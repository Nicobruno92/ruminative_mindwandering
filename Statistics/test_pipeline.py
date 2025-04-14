"""
Test script for LMM-based spatial cluster permutation pipeline.

This script creates synthetic data and runs the pipeline to verify installation
and functionality.
"""

import numpy as np
import pandas as pd
import pickle
import tempfile
import shutil
from pathlib import Path


def create_synthetic_data(
    n_subjects: int = 20,
    n_observations_per_subject: int = 50,
    n_channels: int = 64,
    effect_size: float = 0.5,
    seed: int = 42
) -> tuple:
    """
    Create synthetic EEG power data with embedded spatial effect.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_observations_per_subject : int
        Observations per subject
    n_channels : int
        Number of EEG channels
    effect_size : float
        Magnitude of simulated effect
    seed : int
        Random seed
        
    Returns
    -------
    power_data : np.ndarray
        Synthetic power data (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with subject and predictor variables
    """
    np.random.seed(seed)
    
    n_total = n_subjects * n_observations_per_subject
    
    # Create behavioral data
    subject_ids = np.repeat(range(1, n_subjects + 1), n_observations_per_subject)
    onoff = np.random.randint(0, 2, n_total)  # Binary predictor
    distance = np.random.randn(n_total)  # Continuous predictor
    
    df_behavioral = pd.DataFrame({
        'subject': subject_ids,
        'onoff': onoff,
        'distance': distance
    })
    
    # Create power data with embedded effect
    # Channels 10-15 will have an effect related to 'onoff'
    power_data = np.random.randn(n_total, n_channels)
    
    # Add subject-specific random effects
    for subj in range(1, n_subjects + 1):
        subj_mask = subject_ids == subj
        subj_effect = np.random.randn(n_channels) * 0.3
        power_data[subj_mask, :] += subj_effect
    
    # Add effect in channels 10-15 for onoff predictor
    effect_channels = range(10, 16)
    for ch in effect_channels:
        power_data[onoff == 1, ch] += effect_size
    
    return power_data, df_behavioral


def test_pipeline():
    """Test complete pipeline with synthetic data."""
    print("="*80)
    print("TESTING LMM-BASED SPATIAL CLUSTER PERMUTATION PIPELINE")
    print("="*80)
    print()
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory: {temp_dir}")
    
    try:
        # Step 1: Create synthetic data
        print("\n" + "-"*80)
        print("Step 1: Creating synthetic data")
        print("-"*80)
        
        power_data, df_behavioral = create_synthetic_data(
            n_subjects=10,
            n_observations_per_subject=30,
            n_channels=32,
            effect_size=0.8,
            seed=42
        )
        
        print(f"✓ Power data shape: {power_data.shape}")
        print(f"✓ Behavioral data shape: {df_behavioral.shape}")
        print(f"✓ Number of subjects: {df_behavioral['subject'].nunique()}")
        
        # Save synthetic data
        data_path = Path(temp_dir) / "test_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'power_data': power_data,
                'behavioral_data': df_behavioral
            }, f)
        print(f"✓ Data saved to {data_path}")
        
        # Step 2: Create test configuration
        print("\n" + "-"*80)
        print("Step 2: Creating test configuration")
        print("-"*80)
        
        import yaml
        
        test_config = {
            'project': {
                'data_path': str(data_path),
                'output_path': str(Path(temp_dir) / 'results'),
                'montage_path': 'standard_1020',  # Use standard montage
                'bids_root': '/tmp/test'
            },
            'lmm': {
                'formula': 'power ~ 1 + onoff + (1|subject)',
                'predictor_of_interest': 'onoff',
                'method': 'lbfgs',
                'maxiter': 100,
                'random_state': 42
            },
            'clustering': {
                'threshold': 2.0,
                'n_permutations': 100,  # Small number for testing
                'alpha': 0.05,
                'tail': 0,
                'seed': 42,
                'n_jobs': 1
            },
            'output': {
                'save_pickle': True,
                'save_csv': True,
                'save_figures': True,
                'fig_format': 'png',
                'fig_dpi': 150,
                'overwrite': True
            }
        }
        
        config_path = Path(temp_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        print(f"✓ Configuration saved to {config_path}")
        
        # Step 3: Test individual modules
        print("\n" + "-"*80)
        print("Step 3: Testing individual modules")
        print("-"*80)
        
        # Test reader
        print("\nTesting reader module...")
        from reader import load_data, validate_formula_variables
        
        loaded_power, loaded_behavior = load_data(str(data_path))
        assert loaded_power.shape == power_data.shape
        validate_formula_variables(loaded_behavior, test_config['lmm']['formula'])
        print("✓ Reader module working")
        
        # Test LMM
        print("\nTesting LMM module...")
        from lmm_model import run_lmm_per_channel
        
        t_stats, p_values = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=test_config['lmm']['formula'],
            predictor_of_interest=test_config['lmm']['predictor_of_interest'],
            method=test_config['lmm']['method'],
            maxiter=test_config['lmm']['maxiter'],
            random_state=test_config['lmm']['random_state']
        )
        
        assert t_stats.shape == (32,)
        assert p_values.shape == (32,)
        print(f"✓ LMM module working (t-stats range: [{t_stats.min():.2f}, {t_stats.max():.2f}])")
        
        # Test cluster test
        print("\nTesting cluster test module...")
        from cluster_test import get_channel_adjacency
        
        ch_names = [f'Ch{i+1}' for i in range(32)]
        adjacency, ch_names_ordered = get_channel_adjacency(
            'standard_1020', ch_names[:32]  # Use first 32 channels from standard montage
        )
        print(f"✓ Cluster test module working (adjacency shape: {adjacency.shape})")
        
        # Step 4: Run full pipeline
        print("\n" + "-"*80)
        print("Step 4: Running full pipeline (this may take a minute)")
        print("-"*80)
        
        # Note: We can't easily run the full pipeline without modifying imports
        # so we'll import and call main directly
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        from run_pipeline import main
        
        main(config_path=str(config_path))
        
        # Step 5: Verify outputs
        print("\n" + "-"*80)
        print("Step 5: Verifying outputs")
        print("-"*80)
        
        output_dir = Path(temp_dir) / 'results'
        
        # Check for expected files
        expected_files = [
            'results.pkl',
            't_statistics.csv',
            'cluster_test_topomap.png',
            'cluster_test_distribution.png'
        ]
        
        for fname in expected_files:
            fpath = output_dir / fname
            if fpath.exists():
                print(f"✓ {fname} created")
            else:
                print(f"✗ {fname} missing")
        
        # Load results
        with open(output_dir / 'results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print(f"\n✓ Results loaded successfully")
        print(f"  - Number of clusters: {len(results['clusters'])}")
        print(f"  - T-statistics shape: {results['t_stats'].shape}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print(f"\nTest results saved to: {output_dir}")
        print("You can inspect the outputs to verify the pipeline is working correctly.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        response = input(f"\nDelete temporary directory {temp_dir}? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(temp_dir)
            print("✓ Temporary directory deleted")
        else:
            print(f"Temporary directory preserved at: {temp_dir}")


if __name__ == "__main__":
    test_pipeline()
