import json
import os
import numpy as np

class LogPreprocessingDetails:
    def __init__(self, json_path, subject, task):
        self.json_path = json_path
        self.subject = subject
        self.task = task
        self.logs = self.load_preprocessing_details()

    def load_preprocessing_details(self):
        """
        Load preprocessing details from JSON file.
        
        Returns
        -------
        dict
            Preprocessing logs dictionary, or empty dict if file doesn't
            exist or is corrupted
        """
        if not os.path.exists(self.json_path):
            return {}
        
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"WARNING: JSON file corrupted at {self.json_path}")
            print(f"  Error: {e}")
            print("  Creating backup and starting fresh...")
            
            # Create backup of corrupted file
            backup_path = self.json_path + '.corrupted_backup'
            try:
                import shutil
                shutil.copy2(self.json_path, backup_path)
                print(f"  Backup saved to: {backup_path}")
            except Exception as backup_error:
                print(f"  Could not create backup: {backup_error}")
            
            # Return empty dict to start fresh
            return {}
        except Exception as e:
            print(f"WARNING: Unexpected error loading JSON: {e}")
            print("  Starting with empty log...")
            return {}

    def save_preprocessing_details(self):
        """
        Save preprocessing details to JSON file with atomic write.
        
        Uses temporary file + rename to prevent corruption if process is
        interrupted during write.
        """
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.float64):
                return float(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_logs = convert_to_serializable(self.logs)

        # Atomic write: write to temp file first, then rename
        temp_path = self.json_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(serializable_logs, f, indent=4)
            
            # Atomic rename (replaces old file)
            os.replace(temp_path, self.json_path)
        except Exception as e:
            print(f"ERROR: Failed to save preprocessing log: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise
            
    def initialize_log_structure(self):
        if self.subject not in self.logs:
            self.logs[self.subject] = {}
        if self.task not in self.logs[self.subject]:
            self.logs[self.subject][self.task] = {}

    def log_detail(self, key, value):
        self.initialize_log_structure()
        if isinstance(value, np.ndarray):
            value = value.tolist()  # Convert numpy arrays to lists
        self.logs[self.subject][self.task][key] = value
    
    def clear_subject_task_data(self):
        """Clear data for this specific subject/task to only keep latest run"""
        if self.subject in self.logs and self.task in self.logs[self.subject]:
            del self.logs[self.subject][self.task]

    def get_log(self):
        self.initialize_log_structure()
        return self.logs[self.subject][self.task]
    
    def import_bad_channels_another_task(self):
        self.initialize_log_structure()
        for other_task, details in self.logs[self.subject].items():
            if other_task != self.task and 'interpolated_channels' in details:
                return details['interpolated_channels']
            else:
                return []
