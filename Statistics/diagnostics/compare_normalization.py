#!/usr/bin/env python3
"""
Diagnostic script to compare LMM results with different normalization methods.
Run this to see if "uniform" topographies are caused by z-score normalization.

Usage:
    python Statistics/diagnostics/compare_normalization.py --marker power_gamma
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import yaml
import logging

# Add parent directory to path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from reader import prepare_data_for_lmm
from lmm_model import run_lmm_per_channel
from helpers import normalize_by_subject
from plot_results import validate_montage_and_channels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_probe_data_custom(features_root, subjects=None, tasks=None, marker_types=None, verbose=True):
    """
    Custom loader that handles wide-format state files and prepares them for LMM.
    Target files: sub-XX_task-YY_probe-ZZ_cond_state.csv (wide format)
    """
    features_path = Path(features_root)
    
    # We prioritize _state.csv for this diagnostic task
    patterns = [
        "**/sub-*_task-*_probe-*_*_state.csv"
    ]
    
    csv_files = []
    for p in patterns:
        found = list(features_path.glob(p))
        if found:
            csv_files.extend(found)
            logger.info(f"Found {len(found)} files matching pattern: {p}")
            
    if not csv_files:
        raise ValueError(f"No state CSV files found in {features_root}")
        
    all_data = []
    
    for file_path in csv_files:
        try:
            # Check subject/task filters
            fname = file_path.name
            
            if subjects:
                subj_match = False
                for s in subjects:
                    if f"sub-{s}" in fname:
                        subj_match = True
                        break
                if not subj_match:
                    continue
                    
            if tasks:
                task_match = False
                for t in tasks:
                    if f"task-{t}" in fname:
                        task_match = True
                        break
                if not task_match:
                    continue
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Use 'state' csvs which are wide. We need to normalize columns.
            # Transform wide to long for the LMM
            
            # Map standard markers to columns (based on config.yaml/marker_name_mapping)
            marker_map = {
                'power_delta': 'psd_bands_delta_trimmean',
                'power_theta': 'psd_bands_theta_trimmean',
                'power_alpha': 'psd_bands_alpha_trimmean',
                'power_beta': 'psd_bands_beta_trimmean',
                'power_gamma': 'psd_bands_gamma_trimmean',
                'power_normalized_delta': 'psd_relative_delta_trimmean',
                'power_normalized_theta': 'psd_relative_theta_trimmean',
                'power_normalized_alpha': 'psd_relative_alpha_trimmean',
                'power_normalized_beta': 'psd_relative_beta_trimmean',
                'power_normalized_gamma': 'psd_relative_gamma_trimmean'
            }
            
            # Find which markers are present in columns
            found_markers = []
            for m_name, col_name in marker_map.items():
                if col_name in df.columns:
                    found_markers.append((m_name, col_name))
            
            if not found_markers:
                continue
                
            # Melt to long format
            # Identify ID variables
            id_vars = ['subject', 'task', 'probe_number', 'channel', 'label']
            possible_id_vars = [c for c in id_vars if c in df.columns]
            
            melted = []
            for m_name, col_name in found_markers:
                # Select IDs + value column
                subset = df[possible_id_vars + [col_name]].copy()
                subset = subset.rename(columns={col_name: 'value'})
                subset['marker'] = m_name
                melted.append(subset)
            
            if melted:
                df_long = pd.concat(melted, ignore_index=True)
                
                # Map label to onoff (0-100 scale)
                if 'label' in df_long.columns:
                    df_long['onoff'] = df_long['label'].map({
                        'onTask': 100, 'OnTask': 100, 'ONTASK': 100, 
                        'offTask': 0, 'OffTask': 0, 'OFFTASK': 0
                    })
                
                df_long['marker_type'] = 'state'
                all_data.append(df_long)
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
            
    if not all_data:
        raise ValueError("No valid data loaded")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_data_simple(marker_name, config):
    """Simplified data loader mirroring run_pipeline logic"""
    features_root = config['project']['features_root']
    
    # Load all data using custom loader
    logger.info("Loading all probe data...")
    df_all = load_all_probe_data_custom(
        features_root=features_root,
        subjects=config['project'].get('subjects'),
        tasks=config['project'].get('tasks'),
        marker_types=config['project'].get('marker_types'),
        verbose=True
    )
    
    # Load behavioral data
    behavioral_path = parent_dir.parent / "results" / "Behavior" / "probe_data" / "probe_level_aggregated_data.csv"
    if behavioral_path.exists():
        logger.info(f"Loading behavioral data from {behavioral_path}")
        df_beh_source = pd.read_csv(behavioral_path)
        
        # Standardize merge keys
        # Ensure subject is string and zero-padded if necessary
        # In EEG data, subject is likely numeric or string?
        # Let's inspect df_all['subject'] dtype
        # If numeric, convert to string padded
        
        df_all['subject'] = df_all['subject'].astype(str).str.zfill(2)
        df_beh_source['subject'] = df_beh_source['subject'].astype(str).str.zfill(2)
        
        # Ensure probe_number is int
        df_all['probe_number'] = df_all['probe_number'].astype(int)
        df_beh_source['probe_number'] = df_beh_source['probe_number'].astype(int)
        
        # Select relevant columns to merge
        # onoff is already in df_all mapped from label, but we can take other vars
        cols_to_use = ['subject', 'task', 'probe_number', 'valence', 'selfother', 'time', 'confidence', 'time_on_task']
        cols_to_use = [c for c in cols_to_use if c in df_beh_source.columns]
        
        # Merge
        logger.info(f"Merging behavioral vars: {cols_to_use}")
        df_all = df_all.merge(df_beh_source[cols_to_use], on=['subject', 'task', 'probe_number'], how='left')
        
        # Fill missing?
        # prepare_data_for_lmm handles missing/NaNs by dropping?
        # Or using linear mixed models which handle missing data (unlikely for predictors)
        
    else:
        logger.warning(f"Behavioral data not found at {behavioral_path}")
    
    # Prepare specific marker data
    logger.info(f"Extracting data for {marker_name}...")
    
    formula = config['lmm']['formula']
    predictor = config['lmm']['predictor_of_interest']
    
    logger.info(f"Formula: {formula}")
    logger.info(f"Predictor: {predictor}")

    power_data, df_behavioral, channels = prepare_data_for_lmm(
        df=df_all,
        marker_name=marker_name,
        formula=formula,
        predictor_of_interest=predictor
    )
    
    # Create MNE Info
    info = mne.create_info(ch_names=channels, sfreq=250, ch_types='eeg')
    montage_path = config['project'].get('montage_path')
    if montage_path:
        try:
            if str(montage_path).endswith('.bvef'):
                 montage = mne.channels.read_custom_montage(montage_path)
            else:
                 montage = mne.channels.make_standard_montage(montage_path)
            info.set_montage(montage)
            logger.info(f"Set montage from {montage_path}")
        except Exception as e:
            logger.warning(f"Failed to set montage from config: {e}")
            # Try loading montage file from local path if it's a file path
            local_montage = Path(parent_dir.parent / "BIDS" / "derivatives" / "montage" / Path(montage_path).name)
            if local_montage.exists():
                try:
                    montage = mne.channels.read_custom_montage(str(local_montage))
                    info.set_montage(montage)
                    logger.info(f"Set montage from local file {local_montage}")
                except Exception as e2:
                    logger.warning(f"Failed to set local montage: {e2}")
            
    return power_data, df_behavioral, info

def plot_comparison(results, info, marker_name, predictor):
    logger.info("Generating comparison plots...")
    
    n_scenarios = len(results)
    fig = plt.figure(figsize=(5 * n_scenarios, 12))
    
    # Determine common color limits for t-stats
    all_t = np.concatenate([r["t_stats"] for r in results.values()])
    t_max = np.nanmax(np.abs(all_t))
    vmin, vmax = -t_max, t_max
    
    for i, (name, res) in enumerate(results.items()):
        t_stats = res["t_stats"]
        p_values = res["p_values"]
        n_sig = np.sum(p_values < 0.05) # Uncorrected significant
        
        # Row 1: Topomaps
        ax_topo = plt.subplot(3, n_scenarios, i + 1)
        
        # Validate montage
        info_valid, t_stats_valid = validate_montage_and_channels(info, t_stats)
        t_stats_valid[np.isnan(t_stats_valid)] = 0
        
        mne.viz.plot_topomap(
            t_stats_valid,
            info_valid,
            axes=ax_topo,
            show=False,
            cmap='RdBu_r',
            vlim=(vmin, vmax),
            sensors=False,
            contours=6
        )
        ax_topo.set_title(f"{name}\nSig (uncorr): {n_sig}/{len(t_stats)}")
        
        # Row 2: T-stat Histograms
        ax_hist = plt.subplot(3, n_scenarios, i + 1 + n_scenarios)
        ax_hist.hist(t_stats, bins=20, color='skyblue', edgecolor='black')
        ax_hist.axvline(0, color='k', linestyle='--')
        ax_hist.set_title("T-statistics Distribution")
        ax_hist.set_xlabel("t-value")
        
        # Row 3: Raw Value Distributions (sanity check)
        ax_raw = plt.subplot(3, n_scenarios, i + 1 + 2 * n_scenarios)
        raw_means = res["power_mean"]
        ax_raw.hist(raw_means, bins=20, color='lightgreen', edgecolor='black')
        ax_raw.set_title("Mean Power Distribution")
        ax_raw.set_xlabel("Mean Power")

    plt.suptitle(f"LMM Normalization Comparison - {marker_name} ({predictor})", fontsize=16)
    plt.tight_layout()
    
    output_file = current_dir / f"normalization_comparison_{marker_name}.png"
    plt.savefig(output_file, dpi=150)
    logger.info(f"Comparison plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare LMM normalization methods")
    parser.add_argument("--marker", type=str, default="power_gamma", help="Marker to analyze")
    parser.add_argument("--config", type=str, default="Statistics/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Override paths for local execution if needed
    features_root = Path(config['project']['features_root'])
    if not features_root.exists():
        # Try local path relative to project root (2 levels up from script)
        local_features = parent_dir.parent / "BIDS" / "features"
        if local_features.exists():
            logger.info(f"Config features_root not found. Using local: {local_features}")
            config['project']['features_root'] = str(local_features)
        else:
            logger.error(f"Features root not found: {features_root} or {local_features}")
            
    # 1. Load Data
    logger.info(f"Loading data for marker: {args.marker}")
    try:
        power_data, df_behavioral, info = load_data_simple(args.marker, config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return
        
    if power_data is None:
        logger.error("Failed to load data")
        return
        
    logger.info(f"Loaded data: {power_data.shape} observations, {len(info['ch_names'])} channels")
    
    # Define scenarios
    scenarios = [
        {"name": "Raw (No Norm)", "norm": False, "method": None},
        {"name": "Z-score (Standard)", "norm": True, "method": "zscore"},
        {"name": "Robust (Median/IQR)", "norm": True, "method": "robust"},
        {"name": "MinMax (0-1)", "norm": True, "method": "minmax"}
    ]
    
    results = {}
    
    # LMM settings from config
    formula = config['lmm']['formula']
    predictor = config['lmm']['predictor_of_interest']
    
    logger.info(f"Running LMM with formula: {formula}")
    logger.info(f"Predictor of interest: {predictor}")
    
    # 2. Run Analysis for each scenario
    for scenario in scenarios:
        name = scenario["name"]
        logger.info(f"\n--- Running Scenario: {name} ---")
        
        # Prepare data copy
        current_power = power_data.copy()
        
        # Normalize if requested
        if scenario["norm"]:
            logger.info(f"Applying {scenario['method']} normalization...")
            current_power = normalize_by_subject(
                current_power, 
                df_behavioral, 
                method=scenario["method"],
                channel_wise=config['preprocessing'].get('channel_wise', False)
            )
        
        # Run LMM
        logger.info("Fitting LMMs per channel...")
        t_stats, p_values, diagnostics = run_lmm_per_channel(
            current_power,
            df_behavioral,
            formula,
            predictor,
            maxiter=500 # Faster than full run
        )
        
        results[name] = {
            "t_stats": t_stats,
            "p_values": p_values,
            "power_mean": np.nanmean(current_power, axis=0),
            "diagnostics": diagnostics
        }
        
    # 3. Generate Comparison Plot
    plot_comparison(results, info, args.marker, predictor)

if __name__ == "__main__":
    main()
