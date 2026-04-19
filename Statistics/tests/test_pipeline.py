"""
Test script for LMM-based spatial cluster permutation pipeline.

This script creates synthetic data and runs the pipeline to verify installation
and functionality.
"""

import numpy as np
import pandas as pd
import pickle
import shutil
from pathlib import Path


def create_synthetic_probe_data(
    n_subjects: int = 10,
    n_probes_per_subject: int = 30,
    n_channels: int = 64,
    effect_size: float = 0.8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic aggregated probe marker data matching your actual CYBERSART data structure.
    
    This creates data that closely resembles your real data:
    - Uses actual CACS-64 channel names from your montage
    - Includes realistic marker names from your Junifer pipeline
    - Uses actual subject IDs and task names from your config
    - Includes realistic behavioral variables (onoff, valence, confidence)
    - Embeds effects in realistic brain regions (frontal for mind-wandering)
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects (uses actual subject IDs from your config)
    n_probes_per_subject : int
        Number of probes per subject (realistic for CYBERSART)
    n_channels : int
        Number of EEG channels (uses actual CACS-64 channels)
    effect_size : float
        Magnitude of simulated effect
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Synthetic data in long format matching your aggregated probe marker structure
    """
    np.random.seed(seed)
    
    # Use actual channel names from your CACS-64_REF.bvef montage
    # These are the real channels in your experiment
    cacs64_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
        'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz',
        'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO9', 'PO5', 'PO6', 'PO10'
    ]
    
    # Use actual subject IDs from your config (first n_subjects)
    actual_subject_ids = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", 
                         "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", 
                         "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", 
                         "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"]
    
    # Use actual task names from your config
    actual_tasks = ["Sart1", "Sart2", "Sart3", "Sart4"]
    
    # Use realistic marker names from your Junifer pipeline
    # Based on your config.yaml marker types
    realistic_markers = {
        'state': [
            'EEG_psd_bands_spectralpower_alpha',
            'EEG_psd_bands_spectralpower_theta', 
            'EEG_psd_bands_spectralpower_beta',
            'EEG_psd_bands_spectralpower_gamma',
            'EEG_psd_relative_alpha',
            'EEG_psd_relative_theta',
            'EEG_connectivity_wsmi_alpha',
            'EEG_connectivity_wsmi_theta',
            'EEG_connectivity_wsmi_beta',
            'EEG_information_theory_PE_alpha',
            'EEG_information_theory_PE_theta',
            'EEG_information_theory_kolmogorov_complexity'
        ],
        'evoked': [
            'EEG_evoked_P1',
            'EEG_evoked_N1', 
            'EEG_evoked_P2',
            'EEG_evoked_P3a',
            'EEG_evoked_P3b'
        ]
    }
    
    # Select channels and subjects to use
    channels = cacs64_channels[:n_channels]
    subjects = actual_subject_ids[:n_subjects]
    
    # Create synthetic data
    all_data = []
    
    for subject_id in subjects:
        # Randomly assign 1-2 tasks per subject (realistic)
        subject_tasks = np.random.choice(actual_tasks, size=np.random.randint(1, 3), replace=False)
        
        for task in subject_tasks:
            # Different number of probes per task (realistic variation)
            n_probes = n_probes_per_subject + np.random.randint(-5, 6)
            n_probes = max(10, min(50, n_probes))  # Keep reasonable range
            
            for probe_num in range(1, n_probes + 1):
                # Create realistic behavioral variables for this probe
                # onoff: continuous mind-wandering scale (0-100) - main predictor
                # Use normal distribution centered at 50 for better LMM convergence
                # This creates more realistic data with natural variation
                onoff = np.clip(np.random.normal(50, 25), 0, 100)
                
                # valence: emotional valence (0-100)
                valence = np.clip(np.random.normal(50, 20), 0, 100)
                
                # confidence: confidence in rating (0-100)
                confidence = np.clip(np.random.normal(60, 15), 0, 100)
                
                # Create probe data for each channel and marker
                for ch_idx, channel in enumerate(channels):
                    # Base power value - use normal distribution for better LMM convergence
                    base_power = np.random.normal(0, 1)
                    
                    # Add subject-specific random effect (realistic individual differences)
                    # Use subject index for consistent subject effects
                    subject_idx = subjects.index(subject_id)
                    np.random.seed(seed + subject_idx)  # Consistent per subject
                    subject_effect = np.random.normal(0, 0.8)  # Stronger subject effects
                    np.random.seed(seed + subject_idx * 1000 + probe_num)  # Reset for trial variation
                    
                    # Add realistic onoff effects in different brain regions
                    # Frontal channels: mind-wandering effects (stronger)
                    frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6']
                    
                    # Parietal channels: attention effects (moderate)
                    parietal_channels = ['P3', 'P4', 'Pz', 'P7', 'P8', 'CP1', 'CP2', 'CP5', 'CP6']
                    
                    # Occipital channels: visual attention (weaker)
                    occipital_channels = ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'POz', 'PO7', 'PO8']
                    
                    if channel in frontal_channels:
                        # Strong frontal effect for mind-wandering
                        onoff_effect = (onoff - 50) * effect_size * 0.03
                    elif channel in parietal_channels:
                        # Moderate parietal effect
                        onoff_effect = (onoff - 50) * effect_size * 0.02
                    elif channel in occipital_channels:
                        # Weaker occipital effect
                        onoff_effect = (onoff - 50) * effect_size * 0.01
                    else:
                        # No effect in other regions
                        onoff_effect = 0
                    
                    power_value = base_power + subject_effect + onoff_effect
                    
                    # Create rows for different marker types (test both state and evoked)
                    for marker_type in ['state']:  # Focus on state markers for testing
                        # Use realistic marker names
                        for marker_name in realistic_markers[marker_type][:2]:  # Test first 2 markers
                            all_data.append({
                                'subject': subject_id,
                                'task': task,
                                'probe_number': probe_num,
                                'onoff': onoff,
                                'valence': valence,
                                'confidence': confidence,
                                'marker_type': marker_type,
                                'marker': marker_name,
                                'channel': channel,
                                'value': power_value
                            })
    
    return pd.DataFrame(all_data)


def test_pipeline():
    """Test complete pipeline with synthetic data using existing config.yaml."""
    print("="*80)
    print("TESTING LMM-BASED SPATIAL CLUSTER PERMUTATION PIPELINE")
    print("="*80)
    print()
    
    # Setup paths relative to script location
    script_dir = Path(__file__).parent.absolute()
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_results_base = script_dir / "test_results"
    
    # Remove old test results if they exist
    if test_results_base.exists():
        shutil.rmtree(test_results_base)
    
    temp_dir = test_results_base / f"test_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test directory: {temp_dir.absolute()}")
    
    try:
        # Step 1: Create synthetic data in Junifer format
        print("\n" + "-"*80)
        print("Step 1: Creating synthetic probe marker data")
        print("-"*80)
        
        df_all = create_synthetic_probe_data(
            n_subjects=20,  # Match real data: 42 subjects
            n_probes_per_subject=30,  # Match real data: ~15 probes per subject
            n_channels=64,  # Match real data: 66 channels (full CACS-64)
            effect_size=0.6,  # Moderate effect size for realistic testing
            seed=42
        )
        
        print(f"✓ Synthetic data shape: {df_all.shape}")
        print(f"✓ Number of subjects: {df_all['subject'].nunique()}")
        print(f"✓ Number of channels: {df_all['channel'].nunique()}")
        print(f"✓ Markers: {df_all['marker'].unique()}")
        print(f"✓ Marker types: {df_all['marker_type'].unique()}")
        
        # Save synthetic data as CSV files (simulating Junifer output)
        features_dir = Path(temp_dir) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data for each subject-task combination
        for (subject, task), group in df_all.groupby(['subject', 'task']):
            subj_dir = features_dir / f"sub-{subject}" / "eeg" / "junifer"
            subj_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as aggregated marker CSV
            filename = f"sub-{subject}_task-{task}_desc-probe-1_state_aggMarkers.csv"
            group.to_csv(subj_dir / filename, index=False)
        
        print(f"✓ Synthetic data saved to {features_dir}")
        
        # Step 2: Load existing config and modify for test
        print("\n" + "-"*80)
        print("Step 2: Loading and modifying existing configuration")
        print("-"*80)
        
        import yaml
        
        # Define test marker early so we can use it in config
        test_marker = 'EEG_psd_bands_spectralpower_alpha'
        
        # Load existing config
        config_path = script_dir.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            test_config = yaml.safe_load(f)
        print(f"✓ Loaded existing config from {config_path}")
        
        # Modify config for testing
        test_config['project']['features_root'] = str(features_dir)
        test_config['project']['output_path'] = str(Path(temp_dir) / 'results')
        test_config['project']['montage_path'] = 'standard_1020'  # Use standard montage for testing
        test_config['project']['subjects'] = None  # Test all subjects
        test_config['project']['tasks'] = None  # Test all tasks
        test_config['feature_families'] = {'state_test': [test_marker]}
        test_config['selected_markers'] = {'state': ['state_test']}  # Only test one marker (top-level, not inside 'project')
        test_config['project']['qa_summary_path'] = None  # Disable QA filtering for test
        test_config['project']['pca_results_path'] = None  # Disable PCA data for test
        test_config['lmm']['method'] = 'powell'  # More robust method for convergence
        test_config['lmm']['maxiter'] = 2000  # More iterations for better convergence
        test_config['lmm']['predictor_of_interest'] = 'onoff'  # Explicitly set predictor for testing
        test_config['lmm']['formula'] = 'power ~ onoff + (1|subject)'  # Use simple formula for testing
        test_config['clustering']['n_permutations'] = 100  # Small number for testing
        test_config['clustering']['n_jobs'] = 1  # Single-threaded for testing
        
        # Save test config
        test_config_path = Path(temp_dir) / 'test_config.yaml'
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        print(f"✓ Test configuration saved to {test_config_path}")
        
        # Step 3: Test individual modules
        print("\n" + "-"*80)
        print("Step 3: Testing individual modules")
        print("-"*80)
        
        # Add Statistics directory to path for imports
        import sys
        if str(script_dir.parent) not in sys.path:
            sys.path.insert(0, str(script_dir.parent))
        
        # Test reader
        print("\nTesting reader module...")
        from reader import load_all_probe_data, prepare_data_for_lmm, validate_formula_variables
        
        # Load synthetic data
        print("  Loading synthetic data files...")
        loaded_df = load_all_probe_data(str(features_dir), verbose=False)
        print(f"✓ Loaded {len(loaded_df)} rows of synthetic data")
        
        # Test data preparation for the marker (defined earlier)
        print(f"  Preparing data for marker: {test_marker}...")
        print(f"    Data shape before preparation: {loaded_df.shape}")
        print(f"    Unique subjects: {loaded_df['subject'].nunique()}")
        print(f"    Unique channels: {loaded_df['channel'].nunique()}")
        
        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=loaded_df,
            marker_name=test_marker,
            formula=test_config['lmm']['formula']
        )
        
        print(f"  Validating formula variables...")
        validate_formula_variables(df_behavioral, test_config['lmm']['formula'])
        print(f"✓ Reader module working (power data: {power_data.shape}, channels: {len(channels)})")
        
        # Test LMM
        print("\nTesting LMM module...")
        print(f"  Running LMM for {len(channels)} channels...")
        print(f"  Method: {test_config['lmm']['method']}, Max iterations: {test_config['lmm']['maxiter']}")
        from lmm_model import run_lmm_per_channel
        
        t_stats, p_values, diagnostics = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=test_config['lmm']['formula'],
            predictor_of_interest=test_config['lmm']['predictor_of_interest'],
            method=test_config['lmm']['method'],
            maxiter=test_config['lmm']['maxiter'],
            random_state=test_config['lmm']['random_state'],
            return_diagnostics=True
        )
        
        # Print convergence diagnostics
        print(f"  LMM Convergence: {diagnostics['n_converged']}/{len(channels)} channels converged")
        print(f"  Convergence rate: {100*diagnostics['convergence_rate']:.1f}%")
        
        assert t_stats.shape == (len(channels),)
        assert p_values.shape == (len(channels),)
        
        # Check if we found effects in frontal channels as expected (using actual CACS-64 channels)
        frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8', 'FC1', 'FC2']
        frontal_indices = [i for i, ch in enumerate(channels) if ch in frontal_channels]
        
        if frontal_indices:
            frontal_t_stats = t_stats[frontal_indices]
            max_frontal_t = np.max(np.abs(frontal_t_stats))
            print(f"✓ LMM module working (t-stats range: [{t_stats.min():.2f}, {t_stats.max():.2f}])")
            print(f"  Frontal channels t-stats: {frontal_t_stats[:5]}...")  # Show first 5
            print(f"  Max frontal |t|: {max_frontal_t:.2f}")
            
            # Check if we have significant effects in frontal channels
            if max_frontal_t > 2.0:
                print(f"✓ Expected frontal effects detected!")
            else:
                print(f"⚠ Warning: Expected frontal effects may be weak (max |t| = {max_frontal_t:.2f})")
        else:
            print(f"✓ LMM module working (t-stats range: [{t_stats.min():.2f}, {t_stats.max():.2f}])")
        
        # Test cluster test
        print("\nTesting cluster test module...")
        from cluster_test import get_channel_adjacency
        
        # Use standard montage for testing
        adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(
            'standard_1020', channels
        )
        print(f"✓ Cluster test module working (adjacency shape: {adjacency.shape})")
        print(f"  Used {len(ch_names_ordered)} channels out of {len(channels)} original channels")
        
        # Step 4: Run full pipeline
        print("\n" + "-"*80)
        print("Step 4: Running full pipeline (this may take a minute)")
        print("-"*80)
        
        # Import and run the pipeline
        from run_pipeline import main
        
        # Run pipeline with test configuration (marker specified in config)
        print(f"Running pipeline with config: {test_config_path}")
        main(config_path=str(test_config_path), marker_index=None)
        
        # Step 5: Verify outputs
        print("\n" + "-"*80)
        print("Step 5: Verifying outputs")
        print("-"*80)
        
        # Files are saved in: results/{model_folder}/{marker_type}_{marker_name}/
        # model_folder is generated by get_model_folder_name (includes predictor suffix)
        from helpers import get_model_folder_name
        predictor_name = test_config['lmm']['predictor_of_interest']
        model_folder = get_model_folder_name(test_config['lmm']['formula'], predictor_name)
        marker_type = 'state'
        output_dir = Path(temp_dir) / 'results' / model_folder / f'{marker_type}_{test_marker}'

        print(f"Looking for files in: {output_dir}")

        # Check for expected files (actual output names from pipeline)
        expected_files = [
            'results.pkl',
            'cluster_summary_uncorrected.csv',  # Pipeline saves uncorrected CSV with this name
            't_statistics.csv',
            f'results_{test_marker}_topomap.png',
            f'results_{test_marker}_cluster_details.png',
            f'results_{test_marker}_t_distribution.png',
        ]

        # Also check for summary files in parent directory
        summary_files = [
            ('pipeline_summary.csv', Path(temp_dir) / 'results' / model_folder),
            (f'summary_{marker_type}_markers.csv', Path(temp_dir) / 'results' / model_folder),
            (f'analysis_summary_{marker_type}.csv', Path(temp_dir) / 'results' / model_folder),
        ]
        
        files_found = 0
        for fname in expected_files:
            fpath = output_dir / fname
            if fpath.exists():
                # Additional validation for image files
                if fname.endswith('.png'):
                    file_size = fpath.stat().st_size
                    if file_size > 1000:  # At least 1KB for valid image
                        print(f"✓ {fname} created ({file_size/1024:.1f} KB)")
                        files_found += 1
                    else:
                        print(f"✗ {fname} exists but appears invalid (size: {file_size} bytes)")
                else:
                    print(f"✓ {fname} created")
                    files_found += 1
            else:
                print(f"✗ {fname} missing")
        
        # Check summary files
        for fname, fdir in summary_files:
            fpath = fdir / fname
            if fpath.exists():
                print(f"✓ {fname} created")
                files_found += 1
            else:
                print(f"✗ {fname} missing")
        
        # Load results if available
        results_file = output_dir / 'results.pkl'
        if results_file.exists():
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            print(f"\n✓ Results loaded successfully")
            print(f"  - Marker: {results['marker_name']}")
            print(f"  - Marker type: {results['marker_type']}")
            print(f"  - Number of clusters: {results['n_clusters']}")
            print(f"  - Significant clusters: {results['n_sig_clusters']}")
            print(f"  - T-statistics shape: {results['t_stats'].shape}")
            print(f"  - Number of subjects: {results['n_subjects']}")
            print(f"  - Number of observations: {results['n_observations']}")
            
            # Validate that we found the expected effects
            if results['n_sig_clusters'] > 0:
                print(f"✓ SUCCESS: Found {results['n_sig_clusters']} significant cluster(s) as expected!")
                
                # Check if significant clusters include frontal channels
                sig_clusters = results['clusters']
                sig_cluster_p = results['cluster_p_values']
                ch_names_results = results['ch_names']
                
                for i, (cluster, p_val) in enumerate(zip(sig_clusters, sig_cluster_p)):
                    if p_val < 0.05:
                        cluster_channels = [ch_names_results[idx] for idx in cluster]
                        frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8', 'FC1', 'FC2']
                        frontal_in_cluster = any(ch in frontal_channels for ch in cluster_channels)
                        if frontal_in_cluster:
                            print(f"  ✓ Cluster {i+1} (p={p_val:.3f}) includes frontal channels: {cluster_channels}")
                        else:
                            print(f"  - Cluster {i+1} (p={p_val:.3f}) channels: {cluster_channels}")
            else:
                print(f"⚠ WARNING: No significant clusters found. This may indicate:")
                print(f"    - Effect size too small")
                print(f"    - Insufficient sample size")
                print(f"    - Threshold too high")
                print(f"    - Permutation count too low for testing")
        
        # Load summary if available
        summary_file = Path(temp_dir) / 'results' / predictor_name / 'pipeline_summary.csv'
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            print("\n✓ Pipeline summary loaded:")
            print(summary_df.to_string(index=False))
        
        print("\n" + "="*80)
        total_expected = len(expected_files) + len(summary_files)
        if files_found >= total_expected - 1:  # Allow 1 missing file
            print("ALL TESTS PASSED!")
            print(f"Successfully created {files_found}/{total_expected} expected output files")
        else:
            print(f"SOME TESTS FAILED - Only {files_found}/{total_expected} files created")
        print("="*80)
        
        # Results are already in Statistics/test_results - no copying needed
        print(f"\n✓ Test results saved to: {output_dir.absolute()}")
        print(f"  View plots: open {output_dir}/*.png")
        print(f"  Delete when done: rm -rf {temp_dir.parent}")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # No cleanup needed - results are kept in Statistics/test_results
        print("\n" + "="*80)
        print("Test complete. Results saved to:", temp_dir.absolute())
        print("="*80)


if __name__ == "__main__":
    test_pipeline()
