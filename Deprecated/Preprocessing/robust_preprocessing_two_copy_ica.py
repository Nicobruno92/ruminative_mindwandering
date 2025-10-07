#!/usr/bin/env python3
"""
Robust EEG Preprocessing Pipeline with TWO-COPY ICA Strategy

This script implements a modern, robust preprocessing pipeline following
the mandatory processing order and best practices for EEG data.

CLEAN ARCHITECTURE (v2.0):
- Single Referencing Path: PREP robust average reference (ERP-optimized) OR manual (standard)  
- Single Interpolation Path: AutoReject intelligent (ERP-optimized) OR manual (standard)
- No redundant processing: Each step happens exactly once per pipeline

Processing Order:
1. Load raw data
2. Notch + PREP on full-band raw (PREP handles robust referencing)
3. First-pass AutoReject (optional)
4. Create two synchronized copies (high-pass for ICA, low-pass for analysis)
5. Apply montage to copies (PREP already did referencing)
6. Fit ICA on high-passed copy
7. Label & exclude components (conservative thresholds for ERP-optimized)
8. Apply ICA to low-pass copy
9. Centralized interpolation (manual OR deferred to AutoReject)
10. Low-pass filter, epoch & baseline
11. Final AutoReject on epochs
12. Save outputs & QA report
"""

import os

# 🔧 THREADING OPTIMIZATION: Fix ONNX runtime thread-affinity issues
# Set these before any other imports to prevent BLAS threading conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import subprocess
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Try to import structlog, fall back to standard logging if not available
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    
except ImportError:
    STRUCTLOG_AVAILABLE = False
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create a wrapper to make logging calls compatible
    class LoggerWrapper:
        def __init__(self, logger):
            self._logger = logger
        
        def info(self, msg, **kwargs):
            if kwargs:
                msg = f"{msg} - {json.dumps(kwargs)}"
            self._logger.info(msg)
        
        def error(self, msg, **kwargs):
            if kwargs:
                msg = f"{msg} - {json.dumps(kwargs)}"
            self._logger.error(msg)
        
        def warning(self, msg, **kwargs):
            if kwargs:
                msg = f"{msg} - {json.dumps(kwargs)}"
            self._logger.warning(msg)
    
    logger = LoggerWrapper(logger)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MNE and related imports
import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import AutoReject, get_rejection_threshold
from pyprep import NoisyChannels
from mne_icalabel import label_components

# Import helper functions
from utils.log_preprocessing import LogPreprocessingDetails
from utils.bids_compliance import read_raw_custom, save_raw_bids_compliant, save_epoched_bids, make_bids_basename
from utils.preprocessing_helpers import set_chs_montage
from utils.trigger_correction import TriggerCorrector


class RobustEEGPreprocessor:
    """
    Robust EEG preprocessing pipeline implementing TWO-COPY ICA strategy.
    """
    
    def __init__(self, subject_id: str, task: str, config_path: str = "Preprocessing/configs/config_preprocessing.yaml", memory_efficient: bool = False):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., "02")
        task : str
            Task name (e.g., "Sart1")
        config_path : str
            Path to configuration YAML file
        memory_efficient : bool
            Use memory-efficient processing (reduces memory usage at cost of some speed)
        """
        self.subject_id = subject_id
        self.subject = f"S0{subject_id}"
        self.task = task
        self.data = 'eeg'
        self.memory_efficient = memory_efficient
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Apply memory-efficient overrides
        if self.memory_efficient:
            self._apply_memory_efficient_config()
        
        # Set up paths
        self.setup_paths()
        
        # Set up logging
        self.setup_logging()
        
        # Initialize MNE report
        self.report = mne.Report(
            title=f"Robust Preprocessing sub-{subject_id} for {task}",
            verbose=False
        )
        
        # Set global n_jobs (with error handling for corrupted config)
        try:
            mne.set_config('MNE_USE_NUMBA', 'true')
        except RuntimeError as e:
            if "corrupted" in str(e).lower():
                logger.warning("MNE config file corrupted, recreating it")
                # Delete corrupted config and let MNE recreate it
                import os
                config_path = os.path.expanduser('~/.mne/mne-python.json')
                if os.path.exists(config_path):
                    os.remove(config_path)
                # Try again
                mne.set_config('MNE_USE_NUMBA', 'true')
            else:
                raise
        
        logger.info("Preprocessor initialized", 
                   subject=self.subject_id, 
                   task=self.task,
                   memory_efficient=self.memory_efficient,
                   git_hash=self.get_git_hash(),
                   conda_env=self.get_conda_env())
    
    def setup_paths(self):
        """Set up all necessary paths."""
        self.data_root = self.config['data']['root']
        self.raw_path = os.path.join(self.data_root, self.config['data']['raw_folder'])
        
        # Derivatives folder
        self.derivatives_folder = os.path.join(self.data_root, self.config['data']['derivatives_folder'])
        self.derivative_bids_dir = os.path.join(self.derivatives_folder, f"sub-{self.subject_id}", "eeg")
        os.makedirs(self.derivative_bids_dir, exist_ok=True)
        
        # 🔧 FIX: Set up montage path properly
        self.montage_path = os.path.join("Preprocessing", self.config['montage']['file'])
        
        # ICA save path
        self.ica_path = os.path.join(
            self.derivative_bids_dir, 
            f"sub-{self.subject_id}_task-{self.task}_ica.fif"
        )
        
        # JSON log path
        self.json_path = os.path.join(
            self.derivatives_folder, "logs_preprocessing_details_all_subjects_eeg.json"
        )
    
    def setup_logging(self):
        """Set up preprocessing logging."""
        self.log_preprocessing = LogPreprocessingDetails(
            self.json_path, self.subject_id, self.task
        )
    
    def get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def get_conda_env(self) -> str:
        """Get current conda environment."""
        return os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    
    def _apply_memory_efficient_config(self):
        """Apply memory-efficient configuration overrides."""
        logger.info("🔧 Applying memory-efficient configuration")
        
        # Reduce AutoReject parameters for memory efficiency AND SPEED
        if 'autoreject' in self.config and 'epochs' in self.config['autoreject']:
            ar_config = self.config['autoreject']['epochs']
            # Reduce cross-validation folds to MINIMAL
            ar_config['cv_folds'] = 2  # FASTEST: Only 2 CV folds
            # Use single consensus value for maximum speed
            ar_config['consensus'] = [0.7]  # SINGLE VALUE = no grid search
            # Allow more interpolation but keep single value (eliminates grid search)
            ar_config['n_interpolate'] = [4]  # SINGLE VALUE: Up to 4 channels (no grid search)
            # Use faster threshold method
            ar_config['thresh_method'] = 'random_search'
            # Note: random_search is faster than grid search by default
            logger.info(f"🚀 AutoReject NO GRID SEARCH mode: cv_folds={ar_config['cv_folds']}, consensus={ar_config['consensus']}, n_interpolate={ar_config['n_interpolate']}, method={ar_config['thresh_method']}")
        
        # Store ICA component limit for later application (after raw data is loaded)
        if 'ica' in self.config:
            ica_config = self.config['ica']
            
            # 🔧 CONFIGURABLE: Allow percentage-based or absolute component limits
            max_components_config = self.config.get('memory_efficient', {}).get('max_components', 20)
            
            # Store the configuration for later application
            self._ica_max_components_config = max_components_config
            
            # For now, just log the configuration
            if isinstance(max_components_config, str) and max_components_config.endswith('%'):
                logger.info(f"🔧 Will apply percentage-based ICA limit: {max_components_config}")
            else:
                logger.info(f"🔧 Will apply absolute ICA limit: {max_components_config}")
        
        logger.info("Memory-efficient config applied",
                   autoreject_cv_folds=self.config['autoreject']['epochs']['cv_folds'])
    
    def _apply_ica_component_limits(self):
        """Apply ICA component limits after raw data is loaded."""
        if not hasattr(self, 'raw') or self.raw is None:
            logger.warning("Cannot apply ICA component limits: raw data not loaded")
            return
            
        if 'ica' in self.config and hasattr(self, '_ica_max_components_config'):
            ica_config = self.config['ica']
            max_components_config = self._ica_max_components_config
            
            if isinstance(max_components_config, str) and max_components_config.endswith('%'):
                # Percentage-based limit (e.g., "50%" → 50% of channels)
                try:
                    percentage = float(max_components_config.rstrip('%')) / 100.0
                    # Get number of EEG channels (exclude non-EEG)
                    n_eeg_channels = len([ch for ch in self.raw.ch_names if ch.startswith('EEG')])
                    max_components = max(10, int(n_eeg_channels * percentage))  # Minimum 10 components
                    logger.info(f"🔧 Using percentage-based ICA limit: {percentage*100}% of {n_eeg_channels} EEG channels = {max_components} components")
                except ValueError:
                    max_components = 20  # Fallback
                    logger.warning(f"Invalid percentage format '{max_components_config}', using fallback: {max_components}")
            else:
                # Absolute limit
                max_components = int(max_components_config)
                logger.info(f"🔧 Using absolute ICA limit: {max_components} components")
            
            # Apply the limit
            if isinstance(ica_config.get('n_components'), int):
                ica_config['n_components'] = min(ica_config['n_components'], max_components)
            elif ica_config.get('n_components') is None:
                ica_config['n_components'] = max_components
            
            logger.info(f"ICA component limit applied: {ica_config['n_components']} components")
    
    def _cleanup_memory(self, *object_names):
        """Clean up memory by deleting objects and forcing garbage collection."""
        import gc
        
        # Delete specified object names from self
        for obj_name in object_names:
            if hasattr(self, obj_name):
                delattr(self, obj_name)
                logger.info(f"🔧 Deleted {obj_name} from memory")
        
        # Force garbage collection
        collected = gc.collect()
        
        if self.memory_efficient:
            # Get memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"🔧 Memory cleanup completed, collected {collected} objects, current usage: {memory_mb:.1f} MB")
            except ImportError:
                logger.info(f"🔧 Memory cleanup completed, collected {collected} objects")
    
    def _log_memory_usage(self, step_name: str):
        """Log current memory usage."""
        if self.memory_efficient:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"📊 Memory usage at {step_name}: {memory_mb:.1f} MB")
            except Exception as e:
                logger.info(f"📊 Memory checkpoint: {step_name}", exc_info=True)
    
    def _memory_efficient_load_data(self):
        """Memory-efficient data loading."""
        # Load data normally (read_raw_custom may not support preload=False)
        self.raw = read_raw_custom(self.subject, self.task, root=self.raw_path)
        self.raw = set_chs_montage(self.raw)
        
        # Import bad channels
        self.raw.info["bads"] = self.log_preprocessing.import_bad_channels_another_task()
        
        # For memory efficiency, ensure data is loaded only once
        # Use the correct MNE attribute to check if data is loaded
        if not self.raw.preload:
            logger.info("🔧 Loading data for memory-efficient processing")
            self.raw.load_data()
        
        return self.raw
    
    def _get_n_jobs(self) -> int:
        """Get n_jobs as integer, handling 'auto' and other string values."""
        n_jobs = self.config['computation']['n_jobs']
        if isinstance(n_jobs, str):
            if n_jobs.lower() == 'auto':
                # Use number of CPUs, but cap at reasonable number for stability
                import multiprocessing
                return min(multiprocessing.cpu_count(), 8)
            else:
                try:
                    return int(n_jobs)
                except ValueError:
                    logger.warning(f"Invalid n_jobs value '{n_jobs}', defaulting to 1")
                    return 1
        elif isinstance(n_jobs, int):
            return n_jobs
        else:
            logger.warning(f"Invalid n_jobs type {type(n_jobs)}, defaulting to 1")
            return 1
    
    def load_raw_data(self):
        """Step 1: Load raw data."""
        logger.info("Loading raw data")
        self._log_memory_usage("before_load_raw")
        
        if self.memory_efficient:
            self.raw = self._memory_efficient_load_data()
        else:
            self.raw = read_raw_custom(self.subject, self.task, root=self.raw_path)
            self.raw = set_chs_montage(self.raw)
            # Import bad channels from another task
            self.raw.info["bads"] = self.log_preprocessing.import_bad_channels_another_task()
        
        # 🔧 Apply ICA component limits after raw data is loaded
        if self.memory_efficient:
            self._apply_ica_component_limits()
        
        # Add to report
        self.report.add_raw(raw=self.raw, title="Raw", psd=False)
        
        # Log details
        self.log_preprocessing.log_detail("info", str(self.raw.info))
        
        logger.info("Raw data loaded", 
                   n_channels=len(self.raw.ch_names),
                   duration=self.raw.times[-1],
                   bad_channels=self.raw.info["bads"])
        
        self._log_memory_usage("after_load_raw")
    
    def notch_and_prep(self):
        """Step 2: Notch filtering + PREP on full-band raw."""
        logger.info("Applying notch filtering and PREP")
        self._log_memory_usage("before_notch_prep")
        
        # Store original for comparison
        self.raw0 = self.raw.copy()
        
        # Apply notch filter
        notch_freqs = self.config['filtering']['notch']['freqs']
        self.raw0.load_data().notch_filter(notch_freqs)
        
        logger.info("Notch filtering applied", frequencies=notch_freqs)
        
        # 🔧 OPTIMIZATION: Store imported bad channels and clear them for PREP
        imported_bads = self.raw0.info["bads"].copy()
        logger.info("Storing imported bad channels for PREP", imported_bads=imported_bads)
        self.raw0.info["bads"].clear()
        
        # 🚀 ULTRA-FAST MODE: Use simplified NoisyChannels instead of full PrepPipeline
        prep_config = self.config['prep']
        
        if self.memory_efficient:
            logger.info("🚀 ULTRA-FAST: Using simplified NoisyChannels instead of full PREP for maximum speed")
            
            # Apply montage
            try:
                montage = mne.channels.read_custom_montage(self.montage_path, on_missing='ignore')
            except TypeError:
                logger.info("MNE version doesn't support on_missing parameter, using fallback")
                montage = mne.channels.read_custom_montage(self.montage_path)
            self.raw0.set_montage(montage)
            
            # Simple average reference first
            self.raw0.set_eeg_reference('average', projection=False)
            
            # Use NoisyChannels with speed optimizations
            from pyprep import NoisyChannels
            
            nd = NoisyChannels(
                self.raw0, 
                do_detrend=False, 
                random_state=prep_config['random_state']
            )
            
            # 🚀 ULTRA-FAST: Find bad channels with minimal RANSAC
            nd.find_all_bads(
                ransac=True,  # Keep RANSAC but make it fast
                channel_wise=False  # Skip channel-wise step
            )
            
            prep_bads = nd.get_bads()
            
            # Add detected bad channels
            self.raw0.info["bads"].extend(imported_bads)
            if prep_bads:
                self.raw0.info["bads"].extend(prep_bads)
                self.raw0.info["bads"] = list(set(self.raw0.info["bads"]))  # Remove duplicates
            
            # Interpolate bad channels
            if self.raw0.info["bads"]:
                self.raw0.interpolate_bads(reset_bads=False)
                logger.info("✅ Bad channels interpolated after fast NoisyChannels", channels=self.raw0.info["bads"])
            
            logger.info("✅ Ultra-fast NoisyChannels completed", 
                       prep_bad_channels=prep_bads,
                       imported_bad_channels=imported_bads,
                       total_bad_channels=len(self.raw0.info["bads"]))
            
        else:
            # Standard PREP pipeline for non-ultra-fast mode
            try:
                from pyprep import PrepPipeline
                
                # Apply montage BEFORE PREP to avoid NaN issues
                # 🔧 FIX: Handle MNE version compatibility for on_missing parameter
                try:
                    montage = mne.channels.read_custom_montage(self.montage_path, on_missing='ignore')
                except TypeError:
                    # Older MNE version doesn't support on_missing parameter
                    logger.info("MNE version doesn't support on_missing parameter, using fallback")
                    montage = mne.channels.read_custom_montage(self.montage_path)
                self.raw0.set_montage(montage)
                
                # Set up PREP with speed optimizations for ultra-fast mode
                prep_params = {
                    'ref_chs': 'eeg',         # Use all EEG channels for robust referencing
                    'reref_chs': 'eeg',       # Re-reference all EEG channels  
                    'line_freqs': [50, 100],  # EU line noise frequencies
                }
                
                # 🚀 ULTRA-FAST: Add aggressive RANSAC speed optimizations
                if self.memory_efficient:
                    prep_params.update({
                        'max_iterations': 1,       # Reduce from default 4 to 1 iteration
                        'ransac_window': 0.1,      # Very small windows (much faster RANSAC)
                        'channel_wise_ransac': False,  # Skip channel-wise RANSAC step
                        'matlab_strict': False,    # Don't use MATLAB-strict mode
                        'ransac_fraction_bad': 0.4,  # Allow more bad channels per window (faster convergence)
                        'ransac_n_samples': 50,   # Fewer samples per iteration (much faster)
                    })
                    logger.info("🚀 ULTRA-FAST: Using aggressive RANSAC speed optimizations", 
                               max_iterations=1, ransac_window=0.1, ransac_n_samples=50, 
                               channel_wise_ransac=False, ransac_fraction_bad=0.4)
                
                prep = PrepPipeline(
                    self.raw0,
                    prep_params=prep_params,
                    montage=montage,              # Provide montage to PREP
                    ransac=prep_config.get('ransac', True),
                    random_state=prep_config.get('random_state', 42)
                )
                    
                prep.fit()
                self.raw0 = prep.raw_eeg  # Already re-referenced with robust average
                
                # 🔧 REFERENCE AUDIT: Ensure consistent custom_ref_applied flag
                # PREP leaves robust-average flag, but we need to standardize it
                self.raw0.set_eeg_reference('average', projection=False)
                
                # Collect all bad channels found by PREP
                prep_bads = []
                if hasattr(prep, 'interpolated_channels'):
                    prep_bads.extend(list(prep.interpolated_channels))
                if hasattr(prep, 'bad_channels_original') and 'bad_all' in prep.bad_channels_original:
                    prep_bads.extend(list(prep.bad_channels_original['bad_all']))
                
                # 🔧 OPTIMIZATION: Restore imported bads and add PREP bads
                self.raw0.info["bads"].extend(imported_bads)
                if prep_bads:
                    self.raw0.info["bads"].extend(prep_bads)
                    self.raw0.info["bads"] = list(set(self.raw0.info["bads"]))  # Remove duplicates
                
                # 🔧 OPTIMIZATION: Interpolate bad channels once after PREP
                if self.raw0.info["bads"]:
                    self.raw0.interpolate_bads(reset_bads=False)  # Repair once, let AR handle epoch-wise
                    logger.info("✅ Bad channels interpolated after PREP", channels=self.raw0.info["bads"])
                
                speed_mode = "speed-optimized" if self.memory_efficient else "standard"
                logger.info(f"✅ PREP pipeline completed with robust average reference ({speed_mode})", 
                           prep_bad_channels=prep_bads,
                           imported_bad_channels=imported_bads,
                           total_bad_channels=len(self.raw0.info["bads"]))
                
            except Exception as e:
                logger.warning("PREP pipeline failed, falling back to NoisyChannels only", error=str(e))
                
                # 🔧 OPTIMIZATION: Restore imported bads even in fallback
                self.raw0.info["bads"].extend(imported_bads)
                self.raw0.info["bads"] = list(set(self.raw0.info["bads"]))  # Remove duplicates
                
                # Fallback to original method
                nd = NoisyChannels(
                    self.raw0, 
                    do_detrend=False, 
                    random_state=prep_config['random_state']
                )
                
                # Find bad channels using PREP
                nd.find_all_bads(
                    ransac=prep_config['ransac'], 
                    channel_wise=prep_config['channel_wise']
                )
                
                prep_bads = nd.get_bads()
                if prep_bads:
                    self.raw0.info["bads"].extend(prep_bads)
                    self.raw0.info["bads"] = list(set(self.raw0.info["bads"]))  # Remove duplicates
                
                logger.info("PREP fallback completed", prep_bad_channels=prep_bads)
        
        # 🔧 MEMORY OPTIMIZATION: Clean up original raw data
        if self.memory_efficient:
            self._cleanup_memory('raw')
        
        # Add to report
        self.report.add_raw(raw=self.raw0, title="Raw after Notch + PREP", psd=True)
        
        # Log details
        self.log_preprocessing.log_detail("notch_frequencies", notch_freqs)
        self.log_preprocessing.log_detail("prep_bad_channels", prep_bads)
        self.log_preprocessing.log_detail("imported_bad_channels", imported_bads)
        self.log_preprocessing.log_detail("all_bad_channels", self.raw0.info["bads"])
        
        self._log_memory_usage("after_notch_prep")
    
    def first_pass_autoreject(self):
        """Step 3: First-pass AutoReject on continuous data (optional)."""
        logger.info("Running first-pass AutoReject on continuous data")
        
        # This is optional but recommended for catastrophic segment detection
        # For now, we'll skip this step but log that it could be implemented
        logger.info("First-pass AutoReject skipped (optional step)")
        
        # If implementing, would use autoreject.AutoReject with specific parameters
        # for continuous data and log the results
    
    def create_synchronized_copies(self):
        """
        Step 4: Create two synchronized copies.
        ICA training copy uses YAML ica_training.l_freq (ERP-optimised and standard pipeline).
        """
        logger.info("Creating synchronized copies for TWO-COPY ICA strategy")
        
        # ICA training copy (high-pass filtered)
        ica_config = self.config['filtering']['ica_training']
        logger.info("Creating ICA training copy", ica_config=ica_config)
        self.raw_hp = self.raw0.copy()
        
        # Apply high-pass filter for ICA training (from YAML config)
        logger.info(f"Applying high-pass filter for ICA training ({ica_config['l_freq']} Hz)")
        self.raw_hp.filter(
            l_freq=ica_config['l_freq'], 
            h_freq=ica_config['h_freq']
        )
        
        # Analysis copy (minimal or no high-pass)
        analysis_config = self.config['filtering']['analysis']
        logger.info("Creating analysis copy", analysis_config=analysis_config)
        self.raw_lp = self.raw0.copy()
        
        if analysis_config['l_freq'] is not None:
            logger.info("Applying minimal high-pass filter for analysis")
            self.raw_lp.filter(
                l_freq=analysis_config['l_freq'], 
                h_freq=None
            )
        
        # Guard-rails: verify synchronization
        self._verify_copy_synchronization()
        
        logger.info("Synchronized copies created",
                   ica_filter=f"HP {ica_config['l_freq']} Hz",
                   analysis_filter=f"HP {analysis_config['l_freq']} Hz" if analysis_config['l_freq'] else "No HP")
        
        # Log filter chains
        self.log_preprocessing.log_detail("ica_training_filter", f"notch + HP {ica_config['l_freq']} Hz")
        self.log_preprocessing.log_detail("analysis_filter", "notch only" if analysis_config['l_freq'] is None else f"notch + HP {analysis_config['l_freq']} Hz")
    
    def _verify_copy_synchronization(self):
        """Verify that copies are properly synchronized."""
        # Check same channels
        if self.raw_hp.ch_names != self.raw_lp.ch_names:
            raise ValueError("Channel names differ between copies")
        
        # Check same bad channels
        if set(self.raw_hp.info["bads"]) != set(self.raw_lp.info["bads"]):
            raise ValueError("Bad channels differ between copies")
        
        # Check same reference
        if self.raw_hp.info.get('custom_ref_applied') != self.raw_lp.info.get('custom_ref_applied'):
            logger.warning("Reference status differs between copies")
        
        logger.info("Copy synchronization verified")
    
    def fit_ica(self):
        """Step 5: Fit ICA on high-passed copy."""
        logger.info("Fitting ICA on high-passed copy")
        self._log_memory_usage("before_fit_ica")
        
        ica_config = self.config['ica']
        
        # Initialize ICA
        self.ica = mne.preprocessing.ICA(
            n_components=ica_config['n_components'],
            method=ica_config['method'],
            max_iter=ica_config['max_iter'],
            random_state=ica_config['random_state'],
            fit_params=ica_config['fit_params']
        )
        
        # Fit ICA
        self.ica.fit(self.raw_hp)
        
        # Save ICA
        self.ica.save(self.ica_path, overwrite=True)
        
        logger.info("ICA fitted and saved",
                   n_components=self.ica.n_components_,
                   method=ica_config['method'],
                   path=self.ica_path)
        
        # Log ICA parameters
        self.log_preprocessing.log_detail("ica_n_components", self.ica.n_components_)
        self.log_preprocessing.log_detail("ica_method", ica_config['method'])
        self.log_preprocessing.log_detail("ica_random_state", ica_config['random_state'])
        
        self._log_memory_usage("after_fit_ica")
    
    def label_and_exclude_components(self):
        """Step 6: Label & exclude components."""
        logger.info("Labeling and excluding ICA components")
        
        try:
            # 🔧 OPTIMIZATION: Create temporary 1-100 Hz CAR-referenced copy for ICLabel
            # ICLabel expects CAR-referenced, 1-100 Hz data for optimal performance
            logger.info("Creating ICLabel-optimized temporary copy (1-100 Hz, CAR-referenced)")
            # 🔧 FIX: Extract first element from set_eeg_reference tuple
            tmp_raw = self.raw_hp.copy().filter(1, 100)
            tmp_raw, _ = tmp_raw.set_eeg_reference('average')  # Properly unpack tuple
            
            # ICLabel classification on optimized data
            ic_labels = label_components(tmp_raw, self.ica, method="iclabel")
            
            # Clean up temporary copy immediately
            del tmp_raw
            import gc
            gc.collect()
            
            # 🔧 MEMORY TRACKING: Log memory usage after ICLabel cleanup
            self._log_memory_usage("after_iclabel_tmp")
            
            logger.info("✅ ICLabel completed on optimized data")
            
            # Debug: print available keys and shapes
            logger.info("ICLabel keys available", keys=list(ic_labels.keys()))
            logger.info("ICLabel y_pred_proba shape", shape=ic_labels['y_pred_proba'].shape)
            logger.info("ICLabel labels shape", shape=len(ic_labels['labels']) if hasattr(ic_labels['labels'], '__len__') else 'scalar')
            
            # Save ICLabel scores to CSV
            # Get the class names - they might be in a different key
            if 'classes' in ic_labels:
                class_names = ic_labels['classes']
            elif 'class_names' in ic_labels:
                class_names = ic_labels['class_names']
            else:
                # Default ICLabel class names
                class_names = ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
            
            logger.info("Class names being used", class_names=class_names, n_classes=len(class_names))
            
            # Handle different shapes of y_pred_proba
            y_pred_proba = ic_labels['y_pred_proba']
            if y_pred_proba.ndim == 1:
                # If 1D, we likely have class indices, not probabilities
                logger.warning("ICLabel returned 1D array, treating as class indices")
                # Create a one-hot encoded version
                n_components = len(y_pred_proba)
                n_classes = len(class_names)
                proba_matrix = np.zeros((n_components, n_classes))
                for i, class_idx in enumerate(y_pred_proba):
                    if 0 <= class_idx < n_classes:
                        proba_matrix[i, int(class_idx)] = 1.0
                y_pred_proba = proba_matrix
            elif y_pred_proba.shape[1] == 1:
                # If shape is (n_components, 1), it might be class indices
                logger.warning("ICLabel returned (n_components, 1) array, treating as class indices")
                class_indices = y_pred_proba.flatten()
                n_components = len(class_indices)
                n_classes = len(class_names)
                proba_matrix = np.zeros((n_components, n_classes))
                for i, class_idx in enumerate(class_indices):
                    if 0 <= class_idx < n_classes:
                        proba_matrix[i, int(class_idx)] = 1.0
                y_pred_proba = proba_matrix
            elif y_pred_proba.shape[1] != len(class_names):
                logger.warning("Mismatch between y_pred_proba columns and class names", 
                              proba_cols=y_pred_proba.shape[1], 
                              class_names_len=len(class_names))
                # Adjust class names to match the actual number of columns
                class_names = class_names[:y_pred_proba.shape[1]]
            
            # Create DataFrame safely
            n_cols = min(y_pred_proba.shape[1], len(class_names))
            ic_scores_df = pd.DataFrame(y_pred_proba[:, :n_cols], columns=class_names[:n_cols])
            ic_scores_df['component'] = range(len(ic_scores_df))
            ic_scores_df['predicted_label'] = ic_labels['labels']
            ic_scores_path = os.path.join(
                self.derivative_bids_dir,
                f"sub-{self.subject_id}_task-{self.task}_iclabel-scores.csv"
            )
            ic_scores_df.to_csv(ic_scores_path, index=False)
            
        except Exception as e:
            logger.error("ICLabel classification failed, continuing with pattern matching only", error=str(e))
            # Create dummy variables to continue processing
            y_pred_proba = np.zeros((self.ica.n_components_, 7))
            ic_labels = {'labels': ['other'] * self.ica.n_components_}
        
        # Pattern matching
        pattern_config = self.config['pattern_matching']
        
        # EOG components
        try:
            eog_components, eog_scores = self.ica.find_bads_eog(
                inst=self.raw_hp,
                ch_name=pattern_config['eog']['channels'],
                threshold=pattern_config['eog']['threshold']
            )
        except Exception as e:
            logger.warning("EOG component detection failed", error=str(e))
            eog_components = []
        
        # Muscle components
        try:
            muscle_components, muscle_scores = self.ica.find_bads_muscle(
                self.raw_hp, 
                threshold=pattern_config['muscle']['threshold']
            )
        except Exception as e:
            logger.warning("Muscle component detection failed", error=str(e))
            muscle_components = []
        
        # Combine exclusions
        pattern_matching_artifacts = list(set(eog_components + muscle_components))
        
        # ICLabel-based exclusions (only if ICLabel worked)
        iclabel_exclusions = []
        if ic_labels is not None:
            try:
                iclabel_config = self.config['iclabel']['thresholds']
                
                # Use the processed probability matrix for exclusions
                for i, (label, scores) in enumerate(zip(ic_labels['labels'], y_pred_proba)):
                    label_map = {
                        'eye blink': 'eye_blink',
                        'muscle artifact': 'muscle_artifact', 
                        'heart beat': 'heart_beat',
                        'channel noise': 'channel_noise',
                        'line noise': 'line_noise'
                    }
                    
                    if label in label_map and label_map[label] in iclabel_config:
                        threshold = iclabel_config[label_map[label]]
                        if max(scores) > threshold:
                            iclabel_exclusions.append(i)
            except Exception as e:
                logger.warning("ICLabel-based exclusion failed", error=str(e))
        
        # Final exclusions (union of pattern matching and ICLabel)
        to_exclude = list(set(pattern_matching_artifacts + iclabel_exclusions))
        self.ica.exclude = to_exclude
        
        logger.info("Components excluded",
                   pattern_matching=pattern_matching_artifacts,
                   iclabel_exclusions=iclabel_exclusions,
                   total_excluded=to_exclude)
        
        # Add ICA to report
        self.report.add_ica(self.ica, title="ICA", inst=self.raw_hp)
        
        # Log exclusions
        self.log_preprocessing.log_detail("ica_excluded_components", to_exclude)
        self.log_preprocessing.log_detail("eog_components", eog_components)
        self.log_preprocessing.log_detail("muscle_components", muscle_components)
        self.log_preprocessing.log_detail("iclabel_exclusions", iclabel_exclusions)
    
    def apply_ica_to_analysis_copy(self):
        """Step 7: Apply ICA to low-pass/analysis copy."""
        logger.info("Applying ICA to analysis copy")
        
        # Apply ICA to the analysis copy (in-place)
        self.ica.apply(self.raw_lp, exclude=self.ica.exclude)
        
        logger.info("ICA applied to analysis copy", excluded_components=self.ica.exclude)
    
    def centralized_interpolation(self):
        """
        Centralized bad channel interpolation - happens exactly once.
        Either here for standard pipeline, or deferred to AutoReject for ERP-optimized.
        """
        if hasattr(self, '_prep_handled_referencing'):
            # ERP-optimized path: PREP already handled referencing, skip redundant steps
            logger.info("⏭️ Skipping additional referencing (PREP already handled)")
            logger.info("⏭️ Deferring interpolation to AutoReject for intelligent handling")
            self._interpolation_deferred = True
            return
        
        # Standard pipeline path: manual referencing and interpolation
        logger.info("🔧 Applying standard referencing and interpolation")
        
        # Add reference channels
        ref_channels = self.config['reference']['add_channels']
        self.raw_lp = mne.add_reference_channels(self.raw_lp.load_data(), ref_channels=ref_channels)
        
        # Apply montage
        # 🔧 FIX: Handle MNE version compatibility for on_missing parameter
        try:
            montage = mne.channels.read_custom_montage(self.montage_path, on_missing='ignore')
        except TypeError:
            # Older MNE version doesn't have on_missing parameter
            logger.info("MNE version doesn't support on_missing parameter, using fallback")
            montage = mne.channels.read_custom_montage(self.montage_path)
        self.raw_lp.set_montage(montage)
        
        # Rereference
        ref_method = self.config['reference']['method']
        self.raw_lp, ref_data = mne.set_eeg_reference(
            inst=self.raw_lp, ref_channels=ref_method
        )
        
        # Interpolate bad channels (centralized - happens exactly once)
        if self.raw_lp.info["bads"]:
            self.raw_lp.interpolate_bads()
            logger.info("✅ Bad channels interpolated", channels=self.raw_lp.info["bads"])
        else:
            logger.info("✅ No bad channels to interpolate")
        
        logger.info("Standard referencing and interpolation completed",
                   reference_method=ref_method,
                   interpolated_channels=self.raw_lp.info["bads"])
        
        # Log details
        self.log_preprocessing.log_detail("rereferenced_channels", ref_method)
        self.log_preprocessing.log_detail("interpolated_channels", self.raw_lp.info["bads"])
        self.log_preprocessing.log_detail("interpolation_strategy", "manual_single_pass")
        
        # Add to report
        self.report.add_raw(raw=self.raw_lp, title="Raw rereferenced and interpolated", psd=True)
    
    def lowpass_epoch_baseline(self):
        """Step 8: Low-pass filter, epoch & baseline."""
        logger.info("Applying low-pass filter, epoching, and baseline correction")
        
        # Low-pass filter
        analysis_config = self.config['filtering']['analysis']
        if analysis_config['h_freq'] is not None:
            self.raw_lp.filter(l_freq=None, h_freq=analysis_config['h_freq'])
            logger.info("Low-pass filter applied", frequency=analysis_config['h_freq'])
        
        # Process triggers
        processor = TriggerCorrector(self.raw_lp)
        events, event_id = processor.process_annotations()
        
        # Filter for go/nogo events using regex for precise matching
        import re
        epoch_config = self.config['epoching']
        event_types = epoch_config['event_types']
        
        # Create regex pattern with word boundaries to avoid partial matches
        pattern = r'\b(' + '|'.join(event_types) + r')\b'
        filtered_event_id = {
            key: value for key, value in event_id.items() 
            if re.search(pattern, key, re.IGNORECASE)
        }
        
        # Store for diagnostics
        self.filtered_event_id = filtered_event_id
        
        # Create epochs
        self.epochs = mne.Epochs(
            self.raw_lp,
            events=events,
            event_id=filtered_event_id,
            tmin=epoch_config['tmin'],
            tmax=epoch_config['tmax'],
            baseline=tuple(epoch_config['baseline']),
            preload=True,
            verbose=False,
        )
        
        logger.info("Epoching completed",
                   n_epochs=len(self.epochs),
                   tmin=epoch_config['tmin'],
                   tmax=epoch_config['tmax'],
                   baseline=epoch_config['baseline'])
        
        # Add to report
        self.report.add_epochs(epochs=self.epochs, title="Epochs")
        
        # Log details
        self.log_preprocessing.log_detail("n_epochs", len(self.epochs))
        self.log_preprocessing.log_detail("tmin", epoch_config['tmin'])
        self.log_preprocessing.log_detail("tmax", epoch_config['tmax'])
        self.log_preprocessing.log_detail("baseline", epoch_config['baseline'])
        
        # Store events for saving
        self.events = events
        self.event_id = event_id
    
    def final_autoreject(self):
        """Step 9: Final AutoReject on epochs."""
        logger.info("Running final AutoReject on epochs")
        self._log_memory_usage("before_autoreject")
        
        ar_config = self.config['autoreject']['epochs']
        
        # 🚀 ULTRA-FAST MODE: Use AutoReject with single values (no grid search)
        if self.memory_efficient:
            # Use the optimized config values (single consensus, single interpolation)
            consensus_range = ar_config['consensus']  # Should be [0.7] from config
            n_interpolate_clipped = ar_config['n_interpolate']  # Should be [4] from config
            logger.info("🚀 ULTRA-FAST AutoReject: NO GRID SEARCH (single values only)",
                        consensus=consensus_range, n_interpolate=n_interpolate_clipped,
                        cv_folds=ar_config['cv_folds'], method=ar_config['thresh_method'])
        else:
            # Conservative ERP-optimised settings (override config)
            consensus_range = [0.6, 0.7, 0.8]
            n_interpolate_clipped = [1, 2]
            logger.info("ERP-optimised: Using conservative AutoReject settings",
                        consensus=consensus_range, n_interpolate=n_interpolate_clipped)
        
        ar_params = {
            'n_interpolate': np.array(n_interpolate_clipped),
            'consensus': np.array(consensus_range),
            'thresh_method': ar_config['thresh_method'],
            'cv': ar_config['cv_folds'],
            'random_state': ar_config['random_state'],
            'n_jobs': self._get_n_jobs(),
            'picks': 'eeg'
        }
        
        # 🔧 ULTRA-FAST: Note about random_search optimization
        if self.memory_efficient and ar_config['thresh_method'] == 'random_search':
            # The random_search method in AutoReject will use its default behavior
            # which is already faster than grid search
            logger.info("🚀 ULTRA-FAST: Using random_search method for faster AutoReject processing")
        
        # 🔧 OPTIMIZATION: Ensure epochs have clean channel lists for AutoReject
        if hasattr(self.epochs, 'info') and self.epochs.info.get('bads'):
            logger.info("Clearing bad channels from epochs for AutoReject", 
                       bad_channels=self.epochs.info['bads'])
            self.epochs.info['bads'] = []
        
        # Handle interpolation strategy
        if hasattr(self, '_interpolation_deferred') and self._interpolation_deferred:
            logger.info("🔧 AutoReject will handle ALL interpolation (including pre-marked bad channels)")
        else:
            logger.info("🔧 AutoReject will handle epoch-wise interpolation only")
        
        ar = AutoReject(**ar_params)
        
        # Apply AutoReject
        self.epochs_clean, reject_log = ar.fit_transform(self.epochs, return_log=True)
        
        # Calculate statistics
        n_interpolated_per_epoch = np.sum(reject_log.labels == 1, axis=1)
        mean_interpolated = np.mean(n_interpolated_per_epoch)
        epoch_loss = (1 - len(self.epochs_clean) / len(self.epochs)) * 100
        
        # Warn if targets are not met
        if mean_interpolated > 10:
            logger.warning(f"Mean interpolated channels per epoch is high: {mean_interpolated:.1f} (>10)")
        if epoch_loss > 20:
            logger.warning(f"Epoch loss is high: {epoch_loss:.1f}% (>20%)")
        
        # Get rejection threshold
        reject_threshold = get_rejection_threshold(self.epochs)
        
        # Get actual fitted values from AutoReject (with trailing underscore)
        try:
            consensus_val = ar.consensus_
            if isinstance(consensus_val, dict):
                if 'consensus' in consensus_val:
                    consensus_val = consensus_val['consensus']
                elif 'value' in consensus_val:
                    consensus_val = consensus_val['value']
                else:
                    consensus_val = list(consensus_val.values())[0]
            if hasattr(consensus_val, '__len__') and len(consensus_val) == 1:
                consensus_val = float(consensus_val[0])
            elif hasattr(consensus_val, '__len__'):
                consensus_val = float(np.mean(consensus_val))
            else:
                consensus_val = float(consensus_val)
            n_interpolate_val = ar.n_interpolate_
            if isinstance(n_interpolate_val, dict):
                if 'n_interpolate' in n_interpolate_val:
                    n_interpolate_val = n_interpolate_val['n_interpolate']
                elif 'value' in n_interpolate_val:
                    n_interpolate_val = n_interpolate_val['value']
                else:
                    n_interpolate_val = list(n_interpolate_val.values())[0]
            if hasattr(n_interpolate_val, '__len__') and len(n_interpolate_val) == 1:
                n_interpolate_val = int(n_interpolate_val[0])
            elif hasattr(n_interpolate_val, '__len__'):
                n_interpolate_val = int(np.mean(n_interpolate_val))
            else:
                n_interpolate_val = int(n_interpolate_val)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Could not extract fitted AutoReject parameters, using original values", error=str(e))
            consensus_val = ar.consensus
            if hasattr(consensus_val, '__len__') and len(consensus_val) == 1:
                consensus_val = float(consensus_val[0])
            elif hasattr(consensus_val, '__len__'):
                consensus_val = float(np.mean(consensus_val))
            else:
                consensus_val = float(consensus_val)
            n_interpolate_val = ar.n_interpolate
            if hasattr(n_interpolate_val, '__len__') and len(n_interpolate_val) == 1:
                n_interpolate_val = int(n_interpolate_val[0])
            elif hasattr(n_interpolate_val, '__len__'):
                n_interpolate_val = int(np.mean(n_interpolate_val))
            else:
                n_interpolate_val = int(n_interpolate_val)
        
        logger.info("AutoReject completed",
                   n_epochs_before=len(self.epochs),
                   n_epochs_after=len(self.epochs_clean),
                   mean_interpolated_channels=mean_interpolated,
                   consensus=consensus_val,
                   n_interpolate=n_interpolate_val)
        
        # Log details
        self.log_preprocessing.log_detail("autoreject_consensus", consensus_val)
        self.log_preprocessing.log_detail("autoreject_n_interpolate", n_interpolate_val)
        self.log_preprocessing.log_detail("mean_interpolated_channels", float(mean_interpolated))
        self.log_preprocessing.log_detail("autoreject_threshold", reject_threshold)
        self.log_preprocessing.log_detail("epochs_drop_log", self.epochs_clean.drop_log)
        
        # Save reject log as CSV for embedding in report
        reject_log_df = pd.DataFrame(reject_log.labels, 
                                   columns=self.epochs_clean.ch_names)
        reject_log_df['epoch'] = range(len(reject_log_df))
        reject_log_path = os.path.join(
            self.derivative_bids_dir,
            f"sub-{self.subject_id}_task-{self.task}_reject-log.csv"
        )
        reject_log_df.to_csv(reject_log_path, index=False)
        
        # Add to report
        self.report.add_epochs(epochs=self.epochs_clean, title="Epochs Clean", psd=True)
        
        self._log_memory_usage("after_autoreject")
        
        # Memory cleanup: remove original epochs to save memory
        if self.memory_efficient and hasattr(self, 'epochs'):
            self._cleanup_memory('epochs')
    
    def _simple_threshold_rejection(self):
        """
        Ultra-fast simple threshold rejection as alternative to AutoReject.
        Uses MNE's get_rejection_threshold for speed.
        """
        logger.info("🚀 Running simple threshold rejection (ultra-fast mode)")
        
        # Get simple rejection thresholds
        reject_threshold = get_rejection_threshold(self.epochs)
        
        # Apply simple rejection (no interpolation, just drop bad epochs)
        self.epochs_clean = self.epochs.copy().drop_bad(reject=reject_threshold)
        
        # Calculate statistics
        epoch_loss = (1 - len(self.epochs_clean) / len(self.epochs)) * 100
        
        logger.info("Simple threshold rejection completed",
                   n_epochs_before=len(self.epochs),
                   n_epochs_after=len(self.epochs_clean),
                   epoch_loss_percent=epoch_loss,
                   reject_threshold=reject_threshold)
        
        # Log details
        self.log_preprocessing.log_detail("rejection_method", "simple_threshold")
        self.log_preprocessing.log_detail("reject_threshold", reject_threshold)
        self.log_preprocessing.log_detail("epoch_loss_percent", float(epoch_loss))
        self.log_preprocessing.log_detail("epochs_drop_log", self.epochs_clean.drop_log)
        
        # Add to report
        self.report.add_epochs(epochs=self.epochs_clean, title="Epochs Clean (Simple Rejection)", psd=True)
        
        self._log_memory_usage("after_simple_rejection")
        
        # Memory cleanup: remove original epochs to save memory
        if self.memory_efficient and hasattr(self, 'epochs'):
            self._cleanup_memory('epochs')
    
    def save_outputs_and_report(self):
        """Step 10: Save outputs & QA report."""
        logger.info("Saving outputs and generating QA report")
        
        # Drop bad epochs
        self.epochs_clean.drop_bad()
        
        # Save epoched data
        save_epoched_bids(
            self.epochs_clean,
            self.derivatives_folder,
            self.subject_id,
            self.task,
            self.data,
            desc="autoPreproc",
            events=self.events,
            event_id=self.event_id,
        )
        
        # Create evoked response for report
        try:
            p300_evoked = mne.combine_evoked([
                self.epochs_clean['go/correct'].average(), 
                self.epochs_clean['nogo/correct'].average()
            ], weights=[1, -1])
            self.report.add_evokeds(
                evokeds=[p300_evoked],
                titles=["Evoked P300 Go/Nogo"],
            )
        except KeyError:
            logger.warning("Could not create P300 evoked (go/nogo correct conditions not found)")
        
        # Add version information to report
        version_info = {
            'mne_version': mne.__version__,
            'git_hash': self.get_git_hash(),
            'conda_env': self.get_conda_env(),
            'subject': self.subject_id,
            'task': self.task
        }
        
        # Save report
        html_report_fname = make_bids_basename(
            subject=self.subject_id,
            task=self.task,
            suffix=self.data,
            extension=".html",
            desc="autoPreprocReport",
        )
        report_path = os.path.join(self.derivative_bids_dir, html_report_fname)
        self.report.save(report_path, open_browser=False, overwrite=True)
        
        # Save preprocessing details
        self.log_preprocessing.save_preprocessing_details()
        
        logger.info("Outputs saved",
                   report_path=report_path,
                   epochs_saved=True,
                   preprocessing_log_saved=True)
        
        # Log version information
        for key, value in version_info.items():
            self.log_preprocessing.log_detail(f"version_{key}", value)
        
        self._log_memory_usage("pipeline_completed")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting robust EEG preprocessing pipeline")
        
        try:
            # Execute all steps in order
            self.load_raw_data()                    # Step 1
            self.notch_and_prep()                   # Step 2
            self.first_pass_autoreject()            # Step 3 (optional)
            self.create_synchronized_copies()       # Step 4
            self.fit_ica()                          # Step 5
            self.label_and_exclude_components()     # Step 6
            self.apply_ica_to_analysis_copy()       # Step 7
            self.centralized_interpolation()        # Centralized interpolation step
            self.lowpass_epoch_baseline()           # Step 8
            self.final_autoreject()                 # Step 9
            self.save_outputs_and_report()          # Step 10
            
            logger.info("Preprocessing pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error("Preprocessing pipeline failed", error=str(e), exc_info=True)
            return False

    def diagnose_erp_issues(self):
        """
        Diagnostic function to identify common ERP-damaging issues.
        Call this after preprocessing to check for problems.
        """
        issues = []
        
        # 1. Check ICA exclusions for centro-parietal components
        logger.info("🔍 Diagnosing ICA exclusions for potential P3 sources...")
        
        # Plot properties of excluded components
        if hasattr(self, 'ica') and len(self.ica.exclude) > 0:
            try:
                # Look for excluded components with centro-parietal topographies
                excluded_to_check = self.ica.exclude[:min(4, len(self.ica.exclude))]
                fig = self.ica.plot_properties(self.raw_hp, picks=excluded_to_check, show=False)
                self.report.add_figure(fig, title="🚨 Excluded ICA Components - Check for P3-like topographies")
                
                # Log warning about potential P3 exclusion
                logger.warning("Check excluded components manually", 
                             excluded_components=excluded_to_check,
                             instruction="Look for centro-parietal P3-like topographies")
                issues.append("Potential P3 component exclusion - manual review needed")
                
            except Exception as e:
                logger.warning("Could not plot excluded ICA properties", error=str(e))
        
        # 2. Check reference synchronization
        logger.info("🔍 Checking reference synchronization between ICA training and application...")
        
        if hasattr(self, 'raw_hp') and hasattr(self, 'raw_lp'):
            hp_ref = self.raw_hp.info.get('custom_ref_applied', 'unknown')
            lp_ref = self.raw_lp.info.get('custom_ref_applied', 'unknown')
            
            if hp_ref != lp_ref:
                logger.error("Reference mismatch detected!", 
                           ica_training_ref=hp_ref, 
                           ica_application_ref=lp_ref)
                issues.append(f"Reference mismatch: ICA trained on {hp_ref}, applied to {lp_ref}")
        
        # 3. Check high-pass filter mismatch
        logger.info("🔍 Checking high-pass filter mismatch...")
        
        ica_config = self.config['filtering']['ica_training']
        analysis_config = self.config['filtering']['analysis']
        
        if ica_config['l_freq'] != analysis_config.get('l_freq', 0):
            logger.warning("High-pass filter mismatch detected",
                         ica_hp=ica_config['l_freq'],
                         analysis_hp=analysis_config.get('l_freq', 0))
            issues.append(f"HP mismatch: ICA@{ica_config['l_freq']}Hz → Analysis@{analysis_config.get('l_freq', 0)}Hz")
        
        # 4. Check AutoReject aggressiveness
        logger.info("🔍 Checking AutoReject epoch loss...")
        
        if hasattr(self, 'epochs') and hasattr(self, 'epochs_clean'):
            loss_pct = (1 - len(self.epochs_clean) / len(self.epochs)) * 100
            
            if loss_pct > 40:
                logger.warning("High epoch loss detected", 
                             loss_percentage=loss_pct,
                             epochs_before=len(self.epochs),
                             epochs_after=len(self.epochs_clean))
                issues.append(f"High epoch loss: {loss_pct:.1f}% (>{40}%)")
        
        # 5. Check event IDs
        logger.info("🔍 Checking event ID consistency...")
        
        if hasattr(self, 'filtered_event_id'):
            logger.info("Event counts per condition", event_counts=self.filtered_event_id)
            
            # Check for reasonable trial counts
            for event_name, event_code in self.filtered_event_id.items():
                if hasattr(self, 'epochs_clean'):
                    try:
                        n_trials = len(self.epochs_clean[event_name])
                        if n_trials < 20:
                            logger.warning("Low trial count for ERP", 
                                         condition=event_name, 
                                         trial_count=n_trials)
                            issues.append(f"Low trials for {event_name}: {n_trials} (<20)")
                    except KeyError:
                        logger.warning("Event condition not found in epochs", condition=event_name)
                        issues.append(f"Missing condition: {event_name}")
        
        # Summary report
        if issues:
            logger.error("🚨 ERP SUCCESS ISSUES DETECTED", issues=issues)
            return issues
        else:
            logger.info("✅ No major ERP issues detected")
            return []

    def create_erp_friendly_copies(self):
        """
        Alternative to create_synchronized_copies() that prevents high-pass mismatch.
        Uses minimal high-pass (0.1-0.2 Hz) for both copies to avoid slow ERP distortion.
        <0.5 Hz difference tolerated between ICA and analysis copies.
        """
        logger.info("Creating ERP-friendly synchronized copies (minimal HP mismatch)")
        
        # ICA training copy - use moderate high-pass to preserve slow ERPs
        logger.info("Creating ICA training copy with ERP-friendly high-pass")
        self.raw_hp = self.raw0.copy()
        self.raw_hp.filter(l_freq=0.5, h_freq=None)  # ERP-optimised: 0.5 Hz only
        
        # Analysis copy - minimal high-pass to preserve P300
        logger.info("Creating analysis copy with minimal high-pass")  
        self.raw_lp = self.raw0.copy()
        self.raw_lp.filter(l_freq=0.1, h_freq=None)  # Minimal HP to preserve slow ERPs
        
        # Guard-rails: verify synchronization
        self._verify_copy_synchronization()
        
        # Warn only when near the assert limit
        hp_diff = abs(self.raw_hp.info['highpass'] - self.raw_lp.info['highpass'])
        if hp_diff > 0.5:
            logger.warning(f"High-pass mismatch near limit: raw_hp={self.raw_hp.info['highpass']}, raw_lp={self.raw_lp.info['highpass']} (diff={hp_diff})")
        assert hp_diff < 0.51, f"High-pass mismatch: raw_hp={self.raw_hp.info['highpass']}, raw_lp={self.raw_lp.info['highpass']} (diff={hp_diff})"
        
        logger.info("ERP-friendly copies created",
                   ica_filter="HP 0.5 Hz (ERP-friendly)",
                   analysis_filter="HP 0.1 Hz (P300-preserving)")
        
        # Log filter chains for provenance
        self.log_preprocessing.log_detail("ica_training_filter", "notch + HP 0.5 Hz (ERP-friendly)")
        self.log_preprocessing.log_detail("analysis_filter", "notch + HP 0.1 Hz (P300-preserving)")
        
        self._log_memory_usage("after_create_copies")
        
        # Memory cleanup: we can now remove the original raw data to save memory
        if self.memory_efficient and hasattr(self, 'raw'):
            self._cleanup_memory('raw')

    def conservative_ica_exclusions(self):
        """
        More conservative ICA exclusion logic to preserve potential P3 components.
        """
        logger.info("Applying conservative ICA exclusions to preserve ERPs")
        
        # Pattern matching (keep this as is - it's reliable)
        pattern_config = self.config['pattern_matching']
        
        # EOG components
        try:
            eog_components, eog_scores = self.ica.find_bads_eog(
                inst=self.raw_hp,
                ch_name=pattern_config['eog']['channels'],
                threshold=pattern_config['eog']['threshold']
            )
        except Exception as e:
            logger.warning("EOG component detection failed", error=str(e))
            eog_components = []
        
        # Muscle components - make threshold more conservative
        try:
            muscle_components, muscle_scores = self.ica.find_bads_muscle(
                self.raw_hp, 
                threshold=pattern_config['muscle']['threshold'] + 1.0  # More conservative
            )
        except Exception as e:
            logger.warning("Muscle component detection failed", error=str(e))
            muscle_components = []
        
        # Only exclude clear artifacts from pattern matching
        conservative_exclusions = list(set(eog_components + muscle_components))
        
        # 🔧 OPTIMIZATION: ICLabel with optimized data format (1-100 Hz, CAR-referenced)
        try:
            # Create temporary 1-100 Hz CAR-referenced copy for ICLabel
            logger.info("Creating ICLabel-optimized temporary copy for conservative exclusions")
            # 🔧 FIX: Extract first element from set_eeg_reference tuple
            tmp_raw = self.raw_hp.copy().filter(1, 100)
            tmp_raw, _ = tmp_raw.set_eeg_reference('average')  # Properly unpack tuple
            
            ic_labels = label_components(tmp_raw, self.ica, method="iclabel")
            
            # Clean up temporary copy immediately
            del tmp_raw
            import gc
            gc.collect()
            
            # 🔧 MEMORY TRACKING: Log memory usage after ICLabel cleanup
            self._log_memory_usage("after_iclabel_tmp")
            
            iclabel_exclusions = []
            
            # 🔧 CONSERVATIVE: Only exclude if >90% confidence in artifact class (vs typical 70-80%)
            for i, (label, scores) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                if label in ['eye blink', 'muscle artifact', 'heart beat'] and max(scores) > 0.9:
                    iclabel_exclusions.append(i)
                elif label in ['line noise', 'channel noise'] and max(scores) > 0.95:
                    iclabel_exclusions.append(i)
            
            logger.info("Conservative ICLabel exclusions", 
                       iclabel_exclusions=iclabel_exclusions,
                       criteria="90%+ confidence for artifacts")
                       
        except Exception as e:
            logger.warning("ICLabel failed, using pattern matching only", error=str(e))
            iclabel_exclusions = []
        
        # Final conservative exclusions
        to_exclude = list(set(conservative_exclusions + iclabel_exclusions))
        self.ica.exclude = to_exclude
        
        logger.info("Conservative ICA exclusions applied",
                   pattern_matching=conservative_exclusions,
                   iclabel_exclusions=iclabel_exclusions,
                   total_excluded=to_exclude,
                   exclusion_rate=f"{len(to_exclude)}/{self.ica.n_components_}")
        
        return to_exclude
    
    def collapse_event_conditions(self, events, event_id, factors_to_collapse=None):
        """
        🔧 Helper method to collapse event conditions and avoid 'Low trials < 20' issues.
        
        This addresses the problem where every permutation of on/off × self/other × valence × time
        creates conditions with 0-1 trials each.
        
        Parameters
        ----------
        events : array
            Events array from MNE
        event_id : dict
            Original event_id dictionary
        factors_to_collapse : list, optional
            Factors to collapse (e.g., ['onoff', 'valence', 'time'])
            If None, will collapse to core factors: go/nogo and correct/incorrect
            
        Returns
        -------
        events : array
            Modified events array
        event_id : dict
            Simplified event_id dictionary
        """
        if factors_to_collapse is None:
            # Default: Keep only essential factors for ERP analysis
            factors_to_collapse = ['onoff', 'valence', 'time', 'selfother']
        
        logger.info("🔧 Collapsing event conditions to ensure adequate trial counts",
                   factors_to_collapse=factors_to_collapse,
                   original_conditions=len(event_id))
        
        # Create simplified event mapping
        simplified_event_id = {}
        condition_mapping = {}
        
        for condition_name, code in event_id.items():
            # Extract core factors (go/nogo, correct/incorrect)
            parts = condition_name.split('/')
            
            # Keep go/nogo and correct/incorrect, collapse others
            core_parts = []
            for part in parts:
                if any(keyword in part.lower() for keyword in ['go', 'nogo', 'correct', 'incorrect']):
                    core_parts.append(part)
            
            # Create simplified condition name
            if core_parts:
                simplified_name = '/'.join(core_parts)
            else:
                # Fallback: use first part if no core factors found
                simplified_name = parts[0] if parts else 'unknown'
            
            # Map to simplified condition
            if simplified_name not in simplified_event_id:
                simplified_event_id[simplified_name] = code
                condition_mapping[code] = code
            else:
                # Map this code to the existing simplified condition
                condition_mapping[code] = simplified_event_id[simplified_name]
        
        # Update events array
        new_events = events.copy()
        for i, event in enumerate(events):
            if event[2] in condition_mapping:
                new_events[i, 2] = condition_mapping[event[2]]
        
        logger.info("✅ Event conditions collapsed",
                   original_conditions=len(event_id),
                   simplified_conditions=len(simplified_event_id),
                   condition_mapping=simplified_event_id)
        
        return new_events, simplified_event_id

    def apply_montage_to_copies(self):
        """
        Apply montage to both copies after PREP has done robust referencing.
        PREP handles the referencing, we just need to set the montage for spatial info.
        """
        logger.info("🔧 Applying montage to both copies (PREP already handled referencing)")
        
        # Apply montage to both copies for spatial information
        # 🔧 FIX: Handle MNE version compatibility for on_missing parameter
        try:
            montage = mne.channels.read_custom_montage(self.montage_path, on_missing='ignore')
        except TypeError:
            # Older MNE version doesn't have on_missing parameter
            logger.info("MNE version doesn't support on_missing parameter, using fallback")
            montage = mne.channels.read_custom_montage(self.montage_path)
        self.raw_hp.set_montage(montage)
        self.raw_lp.set_montage(montage)
        
        # Ensure both copies have the same bad channels (from PREP)
        self.raw_hp.info["bads"] = self.raw0.info["bads"].copy()
        self.raw_lp.info["bads"] = self.raw0.info["bads"].copy()
        
        # Verify synchronization
        self._verify_copy_synchronization()
        
        logger.info("✅ Montage applied to both copies",
                   bad_channels_marked=self.raw_hp.info["bads"],
                   referencing_source="PREP robust average reference")
        
        # Log details
        self.log_preprocessing.log_detail("montage_applied", True)
        self.log_preprocessing.log_detail("referencing_source", "PREP_robust_average")
        self.log_preprocessing.log_detail("bad_channels_for_interpolation", self.raw_hp.info["bads"])
        self.log_preprocessing.log_detail("interpolation_strategy", "centralized_autoreject")
        
        # Set flag to indicate PREP handled referencing
        self._prep_handled_referencing = True

    def run_erp_optimized_pipeline(self):
        """
        ERP-optimized preprocessing pipeline that fixes the major issues.
        Use this instead of run_full_pipeline() for better P300 preservation.
        """
        logger.info("🧠 Starting ERP-optimized preprocessing pipeline")
        
        try:
            # Execute steps in corrected order
            self.load_raw_data()                        # Step 1
            self.notch_and_prep()                       # Step 2
            self.first_pass_autoreject()                # Step 3 (optional)
            
            # 🔧 CRITICAL FIX: Create ERP-friendly copies BEFORE ICA fitting to prevent HP mismatch
            self.create_erp_friendly_copies()           # Step 4 (FIXED ORDER)
            
            # 🔧 SIMPLIFIED: Apply montage (PREP already did robust referencing)
            self.apply_montage_to_copies()               # NEW: Montage only, trust PREP referencing
            
            # 🔧 CRITICAL FIX: Fit ICA AFTER creating ERP-friendly copies to prevent mismatch warning
            self.fit_ica()                              # Step 5 (MOVED AFTER COPIES)
            
            # 🔧 CONSERVATIVE: Use less aggressive ICA exclusions
            self.conservative_ica_exclusions()          # Step 6 (FIXED)
            
            self.apply_ica_to_analysis_copy()           # Step 7
            
            # 🔧 CENTRALIZED: Handle interpolation intelligently
            self.centralized_interpolation()            # NEW: Single interpolation strategy
            
            self.lowpass_epoch_baseline()               # Step 8
            self.final_autoreject()                     # Step 9
            
            # 🔍 DIAGNOSTIC: Check for ERP issues
            erp_issues = self.diagnose_erp_issues()     # NEW: Diagnostic
            
            self.save_outputs_and_report()              # Step 10
            
            if erp_issues:
                logger.warning("⚠️ ERP issues detected but pipeline completed", issues=erp_issues)
            else:
                logger.info("✅ ERP-optimized preprocessing completed successfully")
            
            return True, erp_issues
            
        except Exception as e:
            logger.error("ERP-optimized preprocessing failed", error=str(e), exc_info=True)
            return False, []


def main():
    """Main function for command-line execution."""
    if len(sys.argv) < 3:
        print("Usage: python robust_preprocessing_two_copy_ica.py <subject_id> <task> [--erp-optimized] [--memory-efficient] [--ultra-fast]")
        print("Example: python robust_preprocessing_two_copy_ica.py 02 Sart1")
        print("Example: python robust_preprocessing_two_copy_ica.py 02 Sart1 --erp-optimized")
        print("Example: python robust_preprocessing_two_copy_ica.py 02 Sart1 --erp-optimized --memory-efficient")
        print("Example: python robust_preprocessing_two_copy_ica.py 02 Sart1 --ultra-fast")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    task = sys.argv[2]
    use_erp_optimized = '--erp-optimized' in sys.argv
    use_memory_efficient = '--memory-efficient' in sys.argv or '--ultra-fast' in sys.argv
    
    # Ultra-fast mode implies both ERP-optimized and memory-efficient
    if '--ultra-fast' in sys.argv:
        use_erp_optimized = True
    
    # Initialize preprocessor
    preprocessor = RobustEEGPreprocessor(subject_id, task, memory_efficient=use_memory_efficient)
    
    # Run pipeline
    if use_erp_optimized:
        if use_memory_efficient:
            print("🧠💾 Running ERP-optimized + memory-efficient pipeline...")
        else:
            print("🧠 Running ERP-optimized pipeline for better P300 preservation...")
        result = preprocessor.run_erp_optimized_pipeline()
        success, erp_issues = result
        
        if erp_issues:
            print(f"⚠️ ERP issues detected: {erp_issues}")
        else:
            print("✅ No ERP issues detected!")
    else:
        if use_memory_efficient:
            print("💾 Running standard pipeline with memory optimization...")
        else:
            print("Running standard pipeline...")
        success = preprocessor.run_full_pipeline()
        erp_issues = []
    
    # Update status CSV
    derivatives_folder = preprocessor.derivatives_folder
    status_df = pd.DataFrame([{
        'subject': subject_id, 
        'task': task, 
        'data': 'eeg', 
        'status': 'preprocessed' if success else 'failed'
    }])
    
    status_path = os.path.join(derivatives_folder, "preprocessing_status.csv")
    if os.path.exists(status_path):
        existing_df = pd.read_csv(status_path)
        status_df = pd.concat([existing_df, status_df], ignore_index=True)
    
    status_df.to_csv(status_path, index=False)
    
    if success:
        print(f"Preprocessing completed successfully for subject {subject_id}, task {task}")
    else:
        print(f"Preprocessing failed for subject {subject_id}, task {task}")
        sys.exit(1)


if __name__ == "__main__":
    main() 