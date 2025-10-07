#!/usr/bin/env python3
"""
MNE-BIDS-Pipeline Wrapper Script

This script runs the MNE-BIDS-Pipeline for a specific subject and task,
replicating the preprocessing steps from the robust preprocessing pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_custom_config(subject_id: str, task: str, base_config_path: str) -> str:
    """
    Create a custom configuration file for specific subject and task.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "02")
    task : str
        Task name (e.g., "Sart1")
    base_config_path : str
        Path to the base configuration file
        
    Returns
    -------
    str
        Path to the created temporary configuration file
    """
    
    # Read base configuration
    with open(base_config_path, 'r') as f:
        base_config = f.read()
    
    # Create temporary config file
    temp_config_path = f"Preprocessing/temp_config_{subject_id}_{task}.py"
    
    # Custom configuration for this subject/task
    custom_config = f"""
{base_config}

# Override for specific subject and task
subjects = ["{subject_id}"]
task = "{task}"

# Set output directory for this specific run
deriv_root = "/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids"

# Enable specific processing steps for EEG preprocessing
steps = [
    "preprocessing/_01_init",
    "preprocessing/_02_find_empty_room", 
    "preprocessing/_03_data_quality",
    "preprocessing/_04_filtering",
    "preprocessing/_05_ica",
    "preprocessing/_06_epochs",
    "preprocessing/_07_autoreject"
]

# ICA settings specific to this run
ica_l_freq = 1.0  # High-pass for ICA training
ica_eog_threshold = 0.8
ica_algorithm = "infomax"

# Filtering settings
l_freq = 0.1  # Minimal high-pass for analysis
h_freq = 40.0  # Low-pass filter

# Notch filter for line noise
notch_freq = 50

# Epoching parameters
epochs_tmin = -0.3
epochs_tmax = 1.2
baseline = (-0.3, 0.0)

# AutoReject parameters
use_autoreject = True
autoreject_n_interpolate = [1, 4, 8, 16, 32]
autoreject_consensus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
autoreject_cv = 10

# Reference and montage
eeg_reference = "average"

# Parallel processing
n_jobs = 16

# Report settings
log_level = "info"
on_error = "continue"

# Random state for reproducibility
random_state = 42
"""
    
    # Write custom configuration
    with open(temp_config_path, 'w') as f:
        f.write(custom_config)
    
    return temp_config_path

def run_mne_bids_pipeline(config_path: str) -> bool:
    """
    Run the MNE-BIDS-Pipeline with the given configuration.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    
    try:
        # Import here to avoid issues if not installed
        import mne_bids_pipeline
        
        logger.info(f"Running MNE-BIDS-Pipeline with config: {config_path}")
        
        # Run the pipeline using the command line interface
        cmd = [
            "python", "-m", "mne_bids_pipeline",
            config_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600*10  # 10 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("MNE-BIDS-Pipeline completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
            return True
        else:
            logger.error(f"MNE-BIDS-Pipeline failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("MNE-BIDS-Pipeline timed out")
        return False
    except Exception as e:
        logger.error(f"Error running MNE-BIDS-Pipeline: {e}")
        return False

def update_status_csv(subject_id: str, task: str, status: str, derivatives_folder: str):
    """
    Update the preprocessing status CSV file.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    task : str
        Task name
    status : str
        Processing status
    derivatives_folder : str
        Path to derivatives folder
    """
    
    # Create status entry
    status_df = pd.DataFrame([{
        'subject': subject_id,
        'task': task,
        'data': 'eeg',
        'status': status,
        'pipeline': 'mne-bids-pipeline',
        'timestamp': datetime.now().isoformat()
    }])
    
    # Ensure derivatives folder exists
    os.makedirs(derivatives_folder, exist_ok=True)
    
    # Update status file
    status_path = os.path.join(derivatives_folder, 'preprocessing_status.csv')
    
    if os.path.exists(status_path):
        existing_df = pd.read_csv(status_path)
        # Remove any existing entry for this subject/task combination
        mask = (existing_df['subject'] == subject_id) & (existing_df['task'] == task)
        existing_df = existing_df[~mask]
        status_df = pd.concat([existing_df, status_df], ignore_index=True)
    
    status_df.to_csv(status_path, index=False)
    logger.info(f"Status updated: {subject_id}, {task}, {status}")

def cleanup_temp_files(temp_config_path: str):
    """Remove temporary configuration file."""
    try:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            logger.info(f"Cleaned up temporary config: {temp_config_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary config: {e}")

def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def main():
    """Main function for command-line execution."""
    if len(sys.argv) != 3:
        print("Usage: python run_mne_bids_wrapper.py <subject_id> <task>")
        print("Example: python run_mne_bids_wrapper.py 02 Sart1")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    task = sys.argv[2]
    
    logger.info(f"Starting MNE-BIDS-Pipeline preprocessing for subject {subject_id}, task {task}")
    logger.info(f"Git commit: {get_git_hash()}")
    logger.info(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    # Paths
    base_config_path = "Preprocessing/config_mne_bids_pipeline.py"
    derivatives_folder = "/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids"
    
    # Check if base config exists
    if not os.path.exists(base_config_path):
        logger.error(f"Base configuration file not found: {base_config_path}")
        sys.exit(1)
    
    temp_config_path = None
    
    try:
        # Create custom configuration
        temp_config_path = create_custom_config(subject_id, task, base_config_path)
        logger.info(f"Created custom configuration: {temp_config_path}")
        
        # Run the pipeline
        success = run_mne_bids_pipeline(temp_config_path)
        
        # Update status
        status = "preprocessed_mne_bids" if success else "failed_mne_bids"
        update_status_csv(subject_id, task, status, derivatives_folder)
        
        if success:
            logger.info(f"Preprocessing completed successfully for subject {subject_id}, task {task}")
        else:
            logger.error(f"Preprocessing failed for subject {subject_id}, task {task}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        update_status_csv(subject_id, task, "failed_mne_bids", derivatives_folder)
        sys.exit(1)
        
    finally:
        # Clean up temporary files
        if temp_config_path:
            cleanup_temp_files(temp_config_path)

if __name__ == "__main__":
    main() 