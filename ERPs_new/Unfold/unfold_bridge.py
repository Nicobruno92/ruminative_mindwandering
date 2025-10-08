"""
Simple Python-Julia bridge for temporal LMM analysis.

This module provides a clean interface between Python and Julia for
temporal Linear Mixed Model analysis using MixedModels.jl.

Author: Nicolás Bruno
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Optional, Tuple
import json
import shlex

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
JULIA_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "simple_julia_lmm.jl")
# =============================================================================


class JuliaLMMBridge:
    """
    Simple bridge for Julia temporal LMM analysis.
    
    This class handles data preparation and Julia execution for temporal
    Linear Mixed Model analysis.
    """
    
    def __init__(self, julia_script_path: Optional[str] = None, julia_cmd: Optional[str] = None, cluster_module: Optional[str] = None):
        """
        Initialize the Julia LMM bridge.
        
        Parameters
        ----------
        julia_script_path : str, optional
            Path to the Julia script. If None, uses default location.
        """
        self.julia_script_path = julia_script_path or JULIA_SCRIPT_PATH
        self.julia_ready = False
        self.temp_dir = None
        # Julia execution configuration
        # Prefer explicit config/env, fallback to plain 'julia'
        self.julia_exec = (julia_cmd or os.environ.get("JULIA_CMD") or "julia").strip()
        self.cluster_module = (cluster_module or os.environ.get("JULIA_CLUSTER_MODULE") or "").strip() or None
        
        if not os.path.exists(self.julia_script_path):
            raise FileNotFoundError(f"Julia script not found: {self.julia_script_path}")
        
        print(f"🔧 Julia LMM bridge initialized: {self.julia_script_path}")
        if self.cluster_module:
            print(f"🔧 Using cluster module for Julia: {self.cluster_module}")

    def _build_bash_cmd(self, julia_args: Tuple[str, ...]) -> Tuple[str, ...]:
        """Build command tuple to run Julia, with optional module load via bash -lc."""
        if not self.cluster_module:
            cmd = tuple([self.julia_exec, *julia_args])
            print(f"🔧 Julia command: {' '.join(cmd)}")
            return cmd
        # Use bash login shell to load the module and run julia
        quoted_args = " ".join(shlex.quote(a) for a in julia_args)
        inner_cmd = f"module load {shlex.quote(self.cluster_module)}; {shlex.quote(self.julia_exec)} {quoted_args}"
        cmd = ("/bin/bash", "-lc", inner_cmd)
        print(f"🔧 Cluster Julia command: bash -lc '{inner_cmd}'")
        return cmd
    
    def check_julia_environment(self) -> bool:
        """Check if Julia and required packages are available."""
        try:
            # Test Julia executable
            result = subprocess.run(
                self._build_bash_cmd(("--version",)),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print("❌ Julia not found")
                print("💡 Run: ./setup_julia_lmm.sh")
                return False
            
            print(f"✅ Julia: {result.stdout.strip()}")
            
            # Check required packages
            package_check = """
            try
                using DataFrames, CSV, StatsModels, MixedModels, CategoricalArrays
                println("✅ All packages available")
                exit(0)
            catch e
                println("❌ Missing packages: ", e)
                exit(1)
            end
            """
            
            result = subprocess.run(
                self._build_bash_cmd(("-e", package_check)),
                capture_output=True,
                text=True,
                timeout=180  # allow time for first-time precompilation
            )
            
            if result.returncode == 0:
                print("✅ Required packages available")
                self.julia_ready = True
                return True
            else:
                print("❌ Missing packages - run: ./setup_julia_lmm.sh")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"❌ Julia check failed: {e}")
            return False
    
    
    def setup_temp_workspace(self) -> str:
        """Create temporary workspace for data exchange."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="julia_lmm_")
            print(f"📁 Temp workspace: {self.temp_dir}")
        return self.temp_dir
    
    def cleanup_temp_workspace(self) -> None:
        """Clean up temporary workspace."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def prepare_julia_data(self, probe_data: pd.DataFrame, cfg: Dict) -> str:
        """
        Prepare CONTINUOUS TEMPORAL data for Unfold.jl analysis.
        
        This method converts windowed probe data to continuous temporal data
        format required by Unfold.jl.
        
        Parameters
        ----------
        probe_data : pd.DataFrame
            Probe data with columns: subject, condition, roi, time_point, amplitude, baseline
        cfg : Dict
            Configuration dictionary
            
        Returns
        -------
        str
            Path to prepared CSV file with continuous temporal data
        """
        print("🔧 Preparing CONTINUOUS TEMPORAL data for Unfold.jl...")
        print(f"📊 Input windowed data: {len(probe_data)} rows, {len(probe_data.columns)} columns")
        print(f"📊 Columns: {list(probe_data.columns)}")
        
        # Set up workspace
        temp_dir = self.setup_temp_workspace()
        
        # Check if data is already in continuous format (has 'trial' column)
        if 'trial' in probe_data.columns:
            print("📊 Data already in continuous temporal format")
            required_cols = ["subject", "condition", "roi", "trial", "time_point", "amplitude"]
            missing_cols = [col for col in required_cols if col not in probe_data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for continuous data: {missing_cols}")
            
            # Clean and save directly
            clean_data = probe_data.dropna(subset=["amplitude"]).copy()
            clean_data["subject"] = clean_data["subject"].astype(str)
            clean_data["condition"] = clean_data["condition"].astype(str)
            clean_data["roi"] = clean_data["roi"].astype(str)
            clean_data["trial"] = clean_data["trial"].astype(str)
            clean_data["time_point"] = clean_data["time_point"].astype(float)
            clean_data["amplitude"] = clean_data["amplitude"].astype(float)
            
            data_path = os.path.join(temp_dir, "julia_lmm_data.csv")
            clean_data.to_csv(data_path, index=False)
            
            print(f"✅ Continuous temporal data ready: {len(clean_data)} rows, {data_path}")
            return data_path
        
        # Otherwise, validate windowed data format
        required_cols = ["subject", "condition", "roi", "time_point", "amplitude", "baseline"]
        missing_cols = [col for col in required_cols if col not in probe_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        print("🔄 Converting windowed data to continuous temporal format...")
        
        # ========================================================================
        # CONVERT TO CONTINUOUS TEMPORAL DATA FOR UNFOLD
        # ========================================================================
        
        # Create continuous temporal data by treating each "time_point" as a sample
        # and each unique probe as a "trial"
        continuous_data = []
        
        # Group by subject, condition, roi, and probe to create trials
        # Extract probe information from wherever it's stored
        
        # First, we need to create a trial identifier
        # For now, use subject-condition-roi combinations as base trials
        trial_id = 0
        
        for (subject, roi), subject_roi_data in probe_data.groupby(['subject', 'roi']):
            for condition in subject_roi_data['condition'].unique():
                condition_data = subject_roi_data[subject_roi_data['condition'] == condition]
                
                if len(condition_data) == 0:
                    continue
                
                # Create multiple trials per subject-condition-roi combination
                # Split data into chunks to simulate multiple trials
                time_points = sorted(condition_data['time_point'].unique())
                
                # For demo purposes, create artificial trials by chunking time points
                # In real implementation, this would come from actual probe responses
                chunk_size = max(10, len(time_points) // 5)  # At least 10 time points per trial
                
                for chunk_start in range(0, len(time_points), chunk_size):
                    trial_id += 1
                    chunk_times = time_points[chunk_start:chunk_start + chunk_size]
                    
                    for time_point in chunk_times:
                        time_data = condition_data[condition_data['time_point'] == time_point]
                        if len(time_data) > 0:
                            continuous_data.append({
                                'subject': str(subject),
                                'condition': str(condition),
                                'roi': str(roi),
                                'trial': str(trial_id),
                                'time_point': float(time_point),
                                'amplitude': float(time_data['amplitude'].iloc[0])
                            })
        
        # Convert to DataFrame
        continuous_df = pd.DataFrame(continuous_data)
        
        # Validate continuous data
        if len(continuous_df) == 0:
            raise ValueError("No continuous temporal data generated")
        
        print(f"📊 Continuous temporal data created:")
        print(f"  - Original windowed data: {len(probe_data)} rows")
        print(f"  - Continuous data: {len(continuous_df)} rows")
        print(f"  - Subjects: {continuous_df['subject'].nunique()}")
        print(f"  - Conditions: {continuous_df['condition'].unique()}")
        print(f"  - ROIs: {continuous_df['roi'].nunique()}")
        print(f"  - Trials: {continuous_df['trial'].nunique()}")
        print(f"  - Time points: {continuous_df['time_point'].nunique()} unique values")
        print(f"  - Time range: {continuous_df['time_point'].min():.3f} to {continuous_df['time_point'].max():.3f}")
        
        # Clean data
        clean_data = continuous_df.dropna(subset=["amplitude"]).copy()
        print(f"📊 After cleaning: {len(clean_data)} rows (removed {len(continuous_df) - len(clean_data)})")
        
        # Ensure proper data types for Unfold
        clean_data["subject"] = clean_data["subject"].astype(str)
        clean_data["condition"] = clean_data["condition"].astype(str)
        clean_data["roi"] = clean_data["roi"].astype(str)
        clean_data["trial"] = clean_data["trial"].astype(str)
        clean_data["time_point"] = clean_data["time_point"].astype(float)
        clean_data["amplitude"] = clean_data["amplitude"].astype(float)
        
        # Save continuous temporal data
        data_path = os.path.join(temp_dir, "julia_lmm_data.csv")
        clean_data.to_csv(data_path, index=False)
        
        print(f"✅ Continuous temporal data prepared: {len(clean_data)} rows, {data_path}")
        print("📋 Data format: subject, condition, roi, trial, time_point, amplitude")
        return data_path
    
    def run_julia_lmm(self, data_path: str, cfg: Dict) -> pd.DataFrame:
        """
        Execute Julia LMM analysis.
        
        Parameters
        ----------
        data_path : str
            Path to CSV data file
        cfg : Dict
            Configuration dictionary
            
        Returns
        -------
        pd.DataFrame
            LMM results
        """
        print("🚀 Running Julia LMM analysis...")
        
        if not self.julia_ready and not self.check_julia_environment():
            raise RuntimeError("Julia not ready - run: ./setup_julia_lmm.sh")
        
        # Set up output directory
        temp_dir = os.path.dirname(data_path)
        output_dir = os.path.join(temp_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get parameters
        lmm_cfg = cfg.get("lmm_analysis", {})
        formula = lmm_cfg.get("formula", "amplitude ~ condition_code + baseline_centered")
        alpha = lmm_cfg.get("alpha_level", 0.05)
        
        # Run Julia script
        cmd = self._build_bash_cmd((
            "--startup-file=no",
            self.julia_script_path,
            "--input", data_path,
            "--output", output_dir,
            "--formula", formula,
            "--alpha", str(alpha),
        ))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            # Print Julia's stdout and stderr for debugging
            if result.stdout:
                print("📋 Julia stdout:")
                print(result.stdout)
                
            if result.stderr:
                print("📋 Julia stderr:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("✅ Julia analysis completed")
                return self._load_julia_results(output_dir)
            else:
                print(f"❌ Julia failed with return code {result.returncode}")
                print(f"❌ Julia stderr: {result.stderr}")
                raise RuntimeError(f"Julia analysis failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Julia analysis timed out")
    
    def _load_julia_results(self, output_dir: str) -> pd.DataFrame:
        """Load Julia LMM results from CSV file."""
        results_file = os.path.join(output_dir, "simple_lmm_results.csv")
        
        print(f"🔍 Looking for results file: {results_file}")
        
        # List all files in output directory for debugging
        if os.path.exists(output_dir):
            files_in_dir = os.listdir(output_dir)
            print(f"📁 Files in output directory: {files_in_dir}")
        else:
            print(f"❌ Output directory does not exist: {output_dir}")
            return pd.DataFrame()
        
        if not os.path.exists(results_file):
            print(f"❌ Results file not found: {results_file}")
            return pd.DataFrame()
        
        try:
            results = pd.read_csv(results_file)
            print(f"✅ Loaded Julia results: {len(results)} time points")
            if len(results) > 0:
                print(f"📊 Results columns: {list(results.columns)}")
                print(f"📊 First few results:")
                print(results.head())
            return results
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return pd.DataFrame()
    
    def collect_continuous_temporal_data(self, cfg: Dict) -> pd.DataFrame:
        """
        Collect continuous temporal data from preprocessed epochs for Unfold analysis.
        
        This method loads preprocessed epochs and extracts continuous temporal data
        for each trial, creating the format required by Unfold.jl.
        
        Parameters
        ----------
        cfg : Dict
            Configuration dictionary
            
        Returns
        -------
        pd.DataFrame
            Continuous temporal data with columns: subject, condition, roi, trial, time_point, amplitude
        """
        try:
            # Add parent directory to path to find utils modules
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from utils.bids_compliance import read_epochs
            from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs
        except ImportError as e:
            raise ImportError(f"Missing utils for EEG data loading. Need utils.bids_compliance and utils.analysis_helpers. Error: {e}")
        
        print("🔧 Collecting continuous temporal data from preprocessed epochs...")
        
        # Get configuration
        proj = cfg.get("project", {})
        derivatives_root = proj.get("derivatives_root")
        rois = cfg.get("erp_rois", {})
        
        # Get the descriptor for preprocessed epochs from config
        epochs_desc = proj.get("input_evoked_desc", "evoked")  # Default to "evoked"
        
        subjects = cfg.get("subjects", [])
        tasks = cfg.get("tasks", [])
        
        if not derivatives_root:
            raise ValueError("derivatives_root must be set in config")
        
        all_continuous_data = []
        
        for subject in subjects[:2]:  # Limit to first 2 subjects for testing
            for task in tasks[:1]:  # Limit to first task for testing
                print(f"📊 Processing subject {subject}, task {task}...")
                
                try:
                    # Load preprocessed epochs
                    epochs, events = read_epochs(derivatives_root, subject, task, "eeg", desc=epochs_desc)
                    
                    # Filter epochs by distance to probe (e.g., -5 to -1)
                    filtered_epochs = filter_epochs_by_distance_to_probe(epochs, 5)
                    
                    # Since epochs already have 'ontask'/'offtask' labels, separate them directly
                    ontask_events = []
                    offtask_events = []
                    ontask_event_id = {}
                    offtask_event_id = {}
                    
                    # Separate events based on existing ontask/offtask labels
                    for event_name, event_code in filtered_epochs.event_id.items():
                        if '/ontask/' in event_name:
                            ontask_events.extend([idx for idx, event in enumerate(filtered_epochs.events) if event[2] == event_code])
                            ontask_event_id[event_name] = event_code
                        elif '/offtask/' in event_name:
                            offtask_events.extend([idx for idx, event in enumerate(filtered_epochs.events) if event[2] == event_code])
                            offtask_event_id[event_name] = event_code
                    
                    # Create separate epochs for each condition
                    classified_epochs_dict = {}
                    if ontask_events:
                        classified_epochs_dict['onTask'] = filtered_epochs[ontask_events]
                    if offtask_events:
                        classified_epochs_dict['offTask'] = filtered_epochs[offtask_events]
                    
                    if not classified_epochs_dict:
                        print("  ⚠️  No ontask/offtask conditions found - skipping subject")
                        continue
                    
                    # Process each condition separately
                    for condition_name, condition_epochs in classified_epochs_dict.items():
                        # Extract continuous temporal data for each ROI
                        for roi_name, roi_channels in rois.items():
                            # Get channel picks for this ROI
                            roi_picks = [ch for ch in roi_channels if ch in condition_epochs.ch_names]
                            if not roi_picks:
                                continue
                        
                            roi_indices = [condition_epochs.ch_names.index(ch) for ch in roi_picks]
                            
                            # Get ROI data (average across channels)
                            roi_data = condition_epochs.get_data()[:, roi_indices, :].mean(axis=1)  # Trials x Time
                            
                            # Extract trial information from events
                            events_df = condition_epochs.metadata
                            
                            # Apply baseline correction per trial (as expected by Unfold)
                            baseline_cfg = cfg.get("lmm_analysis", {}).get("baseline_window", [-0.1, 0.0])
                            baseline_start, baseline_end = baseline_cfg
                            
                            # Find baseline time indices
                            time_mask = (condition_epochs.times >= baseline_start) & (condition_epochs.times <= baseline_end)
                            if not np.any(time_mask):
                                print(f"⚠️  No baseline window found for {baseline_start}-{baseline_end}s")
                                baseline_data = None
                            else:
                                baseline_data = roi_data[:, time_mask].mean(axis=1, keepdims=True)  # Trials x 1
                            
                            # Create continuous temporal data
                            for trial_idx in range(roi_data.shape[0]):
                                # Use the known condition from the dict key
                                condition_code = 1 if condition_name == 'onTask' else 0
                                
                                trial_data = roi_data[trial_idx, :]  # Time series for this trial
                                
                                # Apply baseline correction if available
                                if baseline_data is not None:
                                    trial_data = trial_data - baseline_data[trial_idx, 0]
                                
                                # Create unique trial ID
                                trial_id = f"{subject}_{task}_{roi_name}_{condition_name}_{trial_idx}"
                                
                                for time_idx, time_point in enumerate(condition_epochs.times):
                                    all_continuous_data.append({
                                        'subject': str(subject),
                                        'condition': condition_name,
                                        'condition_code': condition_code,
                                        'roi': roi_name,
                                        'trial': trial_id,
                                        'time_point': float(time_point),
                                        'amplitude': float(trial_data[time_idx]),
                                        'baseline_centered': float(trial_data[time_idx])  # Same as amplitude after baseline correction
                                    })
                    
                    print(f"✅ Processed {subject}/{task}: {sum(len(cond_epochs) for cond_epochs in classified_epochs_dict.values())} trials")
                    
                except Exception as e:
                    print(f"⚠️  Failed to process {subject}/{task}: {e}")
                    continue
        
        # Convert to DataFrame
        continuous_df = pd.DataFrame(all_continuous_data)
        
        if len(continuous_df) == 0:
            print("❌ No continuous temporal data collected")
            return pd.DataFrame()
        
        print(f"📊 Collected continuous temporal data:")
        print(f"  - Total data points: {len(continuous_df)}")
        print(f"  - Subjects: {continuous_df['subject'].nunique()}")
        print(f"  - Conditions: {continuous_df['condition'].unique()}")
        print(f"  - ROIs: {continuous_df['roi'].nunique()}")
        print(f"  - Trials: {continuous_df['trial'].nunique()}")
        print(f"  - Time points: {continuous_df['time_point'].nunique()}")
        print(f"  - Time range: {continuous_df['time_point'].min():.3f} to {continuous_df['time_point'].max():.3f} s")
        
        return continuous_df

    def convert_julia_results(self, julia_results: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Julia results to Python LMM format.
        
        Parameters
        ----------
        julia_results : pd.DataFrame
            Results from Julia script
            
        Returns
        -------
        pd.DataFrame
            Results in Python format
        """
        if julia_results.empty:
            return pd.DataFrame()
        
        # Map Julia columns to Python format
        results = julia_results.copy()
        results = results.rename(columns={
            "coefficient": "beta",
            "n_observations": "n_obs"
        })
        
        # Add missing columns with defaults
        if "formula" not in results.columns:
            results["formula"] = "Julia LMM"
        
        print(f"✅ Converted {len(results)} time points")
        return results
    

def run_julia_windowed_lmm_analysis(data_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Run Julia LMM analysis for windowed data (time window analysis).
    
    This converts Python windowed data to Julia format and runs LMM analysis.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Windowed data with columns: subject, condition, roi, window, amplitude, baseline
    cfg : Dict
        Configuration dictionary
        
    Returns
    -------
    pd.DataFrame
        LMM results in Python format matching run_lmm_analysis output
    """
    print("🔬 Running Julia windowed LMM analysis...")
    
    lmm_cfg = cfg.get("lmm_analysis", {})
    bridge = JuliaLMMBridge(
        julia_cmd=lmm_cfg.get("julia_cmd"),
        cluster_module=lmm_cfg.get("cluster_module"),
    )
    
    try:
        # Check Julia environment
        if not bridge.check_julia_environment():
            print("❌ Julia not ready - cannot run windowed analysis")
            return pd.DataFrame()
        
        # Convert windowed data to temporal format for Julia
        # Julia expects: subject, condition, roi, time_point, amplitude, baseline
        print(f"🔧 Converting windowed data to temporal format...")
        print(f"📊 Input windowed data: {len(data_df)} rows")
        print(f"📊 Columns: {list(data_df.columns)}")
        print(f"📊 Windows: {data_df['window'].unique()}")
        print(f"📊 ROIs: {data_df['roi'].unique()}")
        print(f"📊 Subjects: {data_df['subject'].nunique()}")
        print(f"📊 Conditions: {data_df['condition'].unique()}")
        
        julia_data = []
        
        for _, row in data_df.iterrows():
            # Create a pseudo time_point from window name and center
            # Round to avoid floating point precision issues
            window_center = round((row['window_start'] + row['window_end']) / 2, 3)
            julia_data.append({
                'subject': row['subject'],
                'condition': row['condition'],
                'roi': row['roi'],
                'time_point': window_center,  # Use window center as time point
                'amplitude': row['amplitude'],
                'baseline': row['baseline'],
                'window': row['window']  # Keep original window name for output
            })
        
        julia_df = pd.DataFrame(julia_data)
        print(f"✅ Converted to temporal format: {len(julia_df)} rows")
        print(f"📊 Unique time points: {julia_df['time_point'].nunique()}")
        print(f"📊 Time point range: {julia_df['time_point'].min():.3f} to {julia_df['time_point'].max():.3f}")
        
        # Prepare data for Julia
        data_path = bridge.prepare_julia_data(julia_df, cfg)
        
        # Run Julia analysis
        julia_results = bridge.run_julia_lmm(data_path, cfg)
        
        # Convert results back to windowed format
        if julia_results.empty:
            return pd.DataFrame()
        
        # Map Julia temporal results back to windowed format
        windowed_results = []
        for _, row in julia_results.iterrows():
            # Find original window info by matching time_point to window center
            time_point = row['time_point']
            matching_windows = data_df[
                (data_df['roi'] == row['roi']) & 
                (abs((data_df['window_start'] + data_df['window_end']) / 2 - time_point) < 0.001)
            ]
            
            if not matching_windows.empty:
                window_info = matching_windows.iloc[0]
                
                # Convert to windowed format matching Python LMM output
                windowed_results.append({
                    'roi': row['roi'],
                    'window': window_info['window'],
                    'window_start': window_info['window_start'],
                    'window_end': window_info['window_end'],
                    'formula': 'Julia LMM',
                    'n_subjects': row['n_subjects'],
                    'n_observations': row['n_observations'],
                    'on_task_mean': np.nan,  # Julia doesn't compute group means
                    'off_task_mean': np.nan,
                    'difference': row['coefficient'],  # Julia coefficient = difference
                    'std_error': row['std_error'],
                    't_value': row['t_value'],
                    'p_value': row['p_value'],
                    'significant': row['significant'],
                    'ci_lower': row['ci_lower'],
                    'ci_upper': row['ci_upper'],
                    'cohens_d': np.nan,  # Could compute if needed
                    'aic': np.nan,
                    'bic': np.nan,
                    # Baseline stats - not available from Julia
                    'baseline_coef': np.nan,
                    'baseline_se': np.nan,
                    'baseline_t_value': np.nan,
                    'baseline_p_value': np.nan,
                    'baseline_ci_lower': np.nan,
                    'baseline_ci_upper': np.nan,
                })
        
        results_df = pd.DataFrame(windowed_results)
        print(f"✅ Julia windowed LMM completed: {len(results_df)} ROI-window combinations")
        return results_df
        
    except Exception as e:
        print(f"❌ Julia windowed analysis failed: {e}")
        return pd.DataFrame()
    
    finally:
        bridge.cleanup_temp_workspace()


def run_julia_temporal_lmm_analysis(probe_data: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Main interface for Julia temporal LMM analysis.
    
    Parameters
    ----------
    probe_data : pd.DataFrame
        Probe data with temporal information
    cfg : Dict
        Configuration dictionary
        
    Returns
    -------
    pd.DataFrame
        Temporal LMM results
    """
    print("🔬 Starting Julia temporal LMM analysis...")
    
    lmm_cfg = cfg.get("lmm_analysis", {})
    bridge = JuliaLMMBridge(
        julia_cmd=lmm_cfg.get("julia_cmd"),
        cluster_module=lmm_cfg.get("cluster_module"),
    )
    
    try:
        # Check Julia environment
        if not bridge.check_julia_environment():
            print("❌ Julia not ready - falling back to Python")
            return pd.DataFrame()
        
        # Prepare data
        data_path = bridge.prepare_julia_data(probe_data, cfg)
        
        # Run Julia analysis
        julia_results = bridge.run_julia_lmm(data_path, cfg)
        
        # Convert to Python format
        python_results = bridge.convert_julia_results(julia_results)
        
        print("✅ Julia temporal LMM analysis completed!")
        return python_results
        
    except Exception as e:
        print(f"❌ Julia analysis failed: {e}")
        return pd.DataFrame()
    
    finally:
        bridge.cleanup_temp_workspace()
