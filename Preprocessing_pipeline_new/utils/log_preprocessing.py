"""
Logging utility for EEG preprocessing pipeline.

Stores preprocessing details (bad channels, ICA components, QA metrics, etc.)
in a JSON file for later analysis and quality control.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class LogPreprocessingDetails:
    """
    Logger for preprocessing details.
    
    Stores preprocessing information in a hierarchical JSON structure:
    subject -> session -> task -> details
    
    Parameters
    ----------
    json_path : str
        Path to JSON file for storing logs
    subject : str
        Subject identifier
    task : str
        Task identifier
    session : str, optional
        Session identifier (default: "default")
    """
    
    def __init__(
        self, 
        json_path: str, 
        subject: str, 
        task: str,
        session: str = "default"
    ) -> None:
        self.json_path = json_path
        self.subject = str(subject)
        self.session = str(session)
        self.task = str(task)
        self.logs = self._load_logs()

    def _load_logs(self) -> Dict[str, Any]:
        """Load existing logs from JSON file."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        return obj

    def save_preprocessing_details(self) -> None:
        """Save logs to JSON file."""
        serializable_logs = self._convert_to_serializable(self.logs)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        
        with open(self.json_path, 'w') as f:
            json.dump(serializable_logs, f, indent=4)

    def _initialize_log_structure(self) -> None:
        """Initialize nested dictionary structure for current subject/session/task."""
        if self.subject not in self.logs:
            self.logs[self.subject] = {}
        if self.session not in self.logs[self.subject]:
            self.logs[self.subject][self.session] = {}
        if self.task not in self.logs[self.subject][self.session]:
            self.logs[self.subject][self.session][self.task] = {}

    def log_detail(self, key: str, value: Any) -> None:
        """
        Log a preprocessing detail.
        
        Parameters
        ----------
        key : str
            Name of the detail (e.g., 'bad_channels', 'ica_excluded')
        value : Any
            Value to store (will be converted to JSON-serializable format)
        """
        self._initialize_log_structure()
        value = self._convert_to_serializable(value)
        self.logs[self.subject][self.session][self.task][key] = value

    def get_log(self) -> Dict[str, Any]:
        """Get log dictionary for current subject/session/task."""
        self._initialize_log_structure()
        return self.logs[self.subject][self.session][self.task]

    def import_bad_channels_another_task(self) -> List[str]:
        """
        Import bad channels from another task in the same session.
        
        Useful for ensuring consistent bad channel handling across tasks.
        
        Returns
        -------
        List[str]
            List of bad channel names from another task, or empty list
        """
        self._initialize_log_structure()
        for other_task, details in self.logs[self.subject][self.session].items():
            if other_task != self.task and 'interpolated_channels' in details:
                return details['interpolated_channels']
        return []
