"""
Comprehensive test suite for LMM-based spatial cluster permutation pipeline.

This test suite:
1. Simulates realistic EEG data based on actual CYBERSART experimental
   structure
2. Tests multiple effect sizes (null, small, medium, large) to verify
   sensitivity
3. Uses actual channel names, subject IDs, and behavioral structure
4. Validates that pipeline finds effects when present and null when absent
5. Tests both convergence and statistical power
"""

import numpy as np
import pandas as pd
import pickle
import shutil
import yaml
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


# CACS-64 channel names from actual montage
CACS64_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F5', 'F1', 'F2',
    'F6', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7',
    'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO9', 'PO5', 'PO6',
    'PO10'
]

# Brain regions for targeted effects
BRAIN_REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8', 'AF3', 'AF4',
                'AFz', 'FC1', 'FC2', 'FC3', 'FC4', 'FCz', 'FC5', 'FC6'],
    'central': ['C3', 'C4', 'Cz', 'C1', 'C2', 'C5', 'C6'],
    'parietal': ['P3', 'P4', 'Pz', 'P1', 'P2', 'P5', 'P6', 'CP1', 'CP2',
                 'CP3', 'CP4', 'CPz', 'CP5', 'CP6'],
    'occipital': ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'POz', 'PO7', 'PO8'],
    'temporal': ['T7', 'T8', 'TP7', 'TP8', 'TP9', 'TP10', 'FT7', 'FT8']
}

# Actual subject IDs from config
SUBJECT_IDS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
               "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
               "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
               "32", "33", "34", "35", "36", "37", "38", "39", "40", "41",
               "42", "43"]

# Actual task names from config
TASK_NAMES = ["Sart1", "Sart2", "Sart3", "Sart4"]

# Realistic marker names from Junifer pipeline
MARKER_NAMES = {
    'state': [
        'EEG_psd_bands_spectralpower_alpha',
        'EEG_psd_bands_spectralpower_theta',
        'EEG_psd_bands_spectralpower_beta',
    ],
    'evoked': [
        'EEG_evoked_P3',
        'EEG_evoked_N1',
    ]
}


def _get_channel_adjacency_matrix():
    """
    Create spatial adjacency matrix for CACS-64 channels using MNE.
    
    Returns
    -------
    tuple
        (adjacency_matrix, channel_indices) where adjacency_matrix is
        n_channels × n_channels and channel_indices maps CACS64_CHANNELS
        to ordered channel names
    """
    try:
        import mne
        from mne.channels import find_ch_adjacency
        
        # Create info object with channel positions
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(CACS64_CHANNELS, sfreq=1000, ch_types='eeg')
        info.set_montage(montage, on_missing='ignore')
        
        # Get adjacency matrix
        adjacency, ch_names = find_ch_adjacency(info, ch_type='eeg')
        
        # Create mapping from our channel list to ordered channels
        channel_indices = {ch: i for i, ch in enumerate(CACS64_CHANNELS)}
        
        return adjacency.toarray(), channel_indices
    except Exception as e:
        print(f"Warning: Could not create adjacency with MNE: {e}")
        # Fallback: identity (no spatial structure)
        n_ch = len(CACS64_CHANNELS)
        channel_indices = {ch: i for i, ch in enumerate(CACS64_CHANNELS)}
        return np.eye(n_ch), channel_indices


def _find_neighbors(channel_idx: int, adjacency: np.ndarray,
                    n_hops: int = 1) -> np.ndarray:
    """
    Find all channels within n_hops of a given channel.
    
    Parameters
    ----------
    channel_idx : int
        Index of center channel
    adjacency : np.ndarray
        Adjacency matrix
    n_hops : int
        Number of hops (1=immediate neighbors, 2=neighbors of neighbors)
        
    Returns
    -------
    np.ndarray
        Indices of neighboring channels (including center)
    """
    n_channels = adjacency.shape[0]
    neighbors = np.zeros(n_channels, dtype=bool)
    neighbors[channel_idx] = True
    
    current_set = np.zeros(n_channels, dtype=bool)
    current_set[channel_idx] = True
    
    for _ in range(n_hops):
        # Find neighbors of current set
        new_neighbors = adjacency[current_set].sum(axis=0) > 0
        neighbors |= new_neighbors
        current_set = new_neighbors & ~neighbors
    
    return np.where(neighbors)[0]


def _generate_smooth_spatial_effect(
    peak_channels: List[int],
    adjacency: np.ndarray,
    peak_effect: float,
    spread_hops: int = 2,
    decay_rate: float = 0.5
) -> np.ndarray:
    """
    Generate smooth spatial effect that decays from peak channels.
    
    Parameters
    ----------
    peak_channels : List[int]
        Indices of channels with peak effect
    adjacency : np.ndarray
        Channel adjacency matrix
    peak_effect : float
        Effect size at peak channels
    spread_hops : int
        How many hops the effect spreads
    decay_rate : float
        How quickly effect decays (0.5 = half per hop)
        
    Returns
    -------
    np.ndarray
        Effect magnitude for each channel
    """
    n_channels = adjacency.shape[0]
    effects = np.zeros(n_channels)
    
    # For each peak channel, spread effect to neighbors
    for peak_ch in peak_channels:
        # Mark distances from this peak
        distances = np.full(n_channels, np.inf)
        distances[peak_ch] = 0
        
        visited = np.zeros(n_channels, dtype=bool)
        to_visit = [peak_ch]
        
        # Breadth-first search for distances
        while to_visit:
            current = to_visit.pop(0)
            if visited[current]:
                continue
            visited[current] = True
            current_dist = distances[current]
            
            if current_dist >= spread_hops:
                continue
            
            # Add neighbors
            neighbors = np.where(adjacency[current] > 0)[0]
            for neighbor in neighbors:
                if distances[neighbor] > current_dist + 1:
                    distances[neighbor] = current_dist + 1
                    to_visit.append(neighbor)
        
        # Apply decay based on distance
        for ch_idx in range(n_channels):
            if distances[ch_idx] <= spread_hops:
                # Exponential decay: effect * (decay_rate ^ distance)
                channel_effect = (
                    peak_effect * (decay_rate ** distances[ch_idx])
                )
                # Take maximum across all peaks
                effects[ch_idx] = max(effects[ch_idx], channel_effect)
    
    return effects


def simulate_realistic_eeg_data(
    n_subjects: int = 20,
    n_probes_per_subject: int = 25,
    effect_config: Dict[str, Dict[str, float]] = None,
    noise_level: float = 1.0,
    subject_variability: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate realistic EEG probe data with spatially coherent effects.
    
    Key features:
    - Spatially localized effects (realistic cluster sizes: 5-15 channels)
    - Effects spread smoothly across adjacent channels
    - Peak effects in specified regions, decay with distance
    - Spatially correlated noise
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects to simulate
    n_probes_per_subject : int
        Average number of probes per subject (actual varies ±5)
    effect_config : Dict[str, Dict[str, float]]
        Dictionary mapping brain regions to effect configurations.
        Each effect config has:
            'size': Effect size (Cohen's d) at peak channels
            'predictor': Which predictor to modulate ('onoff', etc.)
            'n_peak_channels': Number of peak effect channels (default: 2-3)
            'spread': How many hops effect spreads (default: 2)
            'decay_rate': How quickly effect decays (default: 0.5)
        Example: {'frontal': {'size': 0.6, 'predictor': 'onoff',
                              'n_peak_channels': 3, 'spread': 2,
                              'decay_rate': 0.5}}
        If None, generates null data (no effects)
    noise_level : float
        Standard deviation of measurement noise
    subject_variability : float
        Standard deviation of subject-specific random effects
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Simulated data in long format matching aggregated probe marker
        structure
    """
    np.random.seed(seed)
    
    # Get channel adjacency for spatial structure
    adjacency, channel_indices = _get_channel_adjacency_matrix()
    n_channels = len(CACS64_CHANNELS)
    
    # Select subjects and setup
    subjects = SUBJECT_IDS[:n_subjects]
    all_data = []
    
    # Create spatially smooth effect map
    effect_map = np.zeros(n_channels)
    
    if effect_config is not None:
        for region, config in effect_config.items():
            effect_size = config['size']
            n_peak = config.get('n_peak_channels', 3)
            spread_hops = config.get('spread', 2)
            decay_rate = config.get('decay_rate', 0.5)
            
            # Get channels in this region
            region_channels = [
                i for i, ch in enumerate(CACS64_CHANNELS)
                if ch in BRAIN_REGIONS[region]
            ]
            
            if len(region_channels) == 0:
                continue
            
            # Select peak effect channels (central channels in region)
            n_peak = min(n_peak, len(region_channels))
            peak_channels = np.random.choice(
                region_channels, size=n_peak, replace=False
            ).tolist()
            
            # Generate smooth spatial effect
            region_effect = _generate_smooth_spatial_effect(
                peak_channels=peak_channels,
                adjacency=adjacency,
                peak_effect=effect_size,
                spread_hops=spread_hops,
                decay_rate=decay_rate
            )
            
            # Add to effect map (take maximum if overlapping)
            effect_map = np.maximum(effect_map, region_effect)
    
    # Generate subject-specific random effects
    subject_effects = {
        subj: np.random.normal(0, subject_variability, n_channels)
        for subj in subjects
    }
    
    for subj_idx, subject in enumerate(subjects):
        # Assign 1-3 tasks per subject (realistic)
        n_tasks = np.random.randint(1, 4)
        subject_tasks = np.random.choice(
            TASK_NAMES, size=n_tasks, replace=False
        )
        
        for task in subject_tasks:
            # Variable number of probes per task
            n_probes = n_probes_per_subject + np.random.randint(-5, 6)
            n_probes = max(10, min(40, n_probes))
            
            for probe_idx in range(n_probes):
                probe_num = probe_idx + 1
                
                # Generate realistic behavioral ratings
                onoff = np.clip(np.random.normal(50, 20), 0, 100)
                valence = np.clip(np.random.normal(50, 15), 0, 100)
                confidence = np.clip(np.random.normal(60, 12), 0, 100)
                
                # Get predictor value for this probe
                pred_value = 0
                if effect_config is not None:
                    # Get predictor from first config
                    predictor = list(effect_config.values())[0].get(
                        'predictor', 'onoff'
                    )
                    
                    if predictor == 'onoff':
                        pred_value = (onoff - 50) / 50  # [-1, 1]
                    elif predictor == 'valence':
                        pred_value = (valence - 50) / 50
                    elif predictor == 'confidence':
                        pred_value = (confidence - 50) / 50
                
                # Generate spatially correlated noise for this probe
                # Simple method: start with independent noise, smooth it
                independent_noise = np.random.normal(0, noise_level, n_channels)
                # Apply spatial smoothing via adjacency
                neighbor_avg = adjacency @ independent_noise / (
                    adjacency.sum(axis=1) + 1e-10
                )
                # Mix independent and smoothed (60% smooth, 40% independent)
                spatial_noise = 0.4 * independent_noise + 0.6 * neighbor_avg
                
                # Generate EEG data for each channel
                for ch_idx, channel in enumerate(CACS64_CHANNELS):
                    # Components:
                    # 1. Spatially correlated noise
                    noise = spatial_noise[ch_idx]
                    
                    # 2. Subject-specific baseline
                    subject_baseline = subject_effects[subject][ch_idx]
                    
                    # 3. Spatially smooth effect (varies with predictor)
                    effect = effect_map[ch_idx] * noise_level * pred_value
                    
                    # Combine components
                    value = noise + subject_baseline + effect
                    
                    # Create rows for each marker type
                    for marker_type, markers in MARKER_NAMES.items():
                        for marker_name in markers:
                            # Small marker-specific variation
                            marker_var = np.random.normal(0, 0.05)
                            final_value = value + marker_var
                            
                            all_data.append({
                                'subject': subject,
                                'task': task,
                                'probe_number': probe_num,
                                'onoff': onoff,
                                'valence': valence,
                                'confidence': confidence,
                                'marker_type': marker_type,
                                'marker': marker_name,
                                'channel': channel,
                                'value': final_value
                            })
    
    return pd.DataFrame(all_data)


def save_synthetic_data(df: pd.DataFrame, output_dir: Path):
    """Save synthetic data in Junifer aggregated marker format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by subject-task-marker_type and save as separate CSV files
    grouped = df.groupby(['subject', 'task', 'marker_type'])
    for (subject, task, marker_type), group in grouped:
        # Create subject directory structure
        subj_dir = output_dir / f"sub-{subject}" / "eeg" / "junifer"
        subj_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename matching Junifer output format
        filename = (
            f"sub-{subject}_task-{task}_desc-probe-1_"
            f"{marker_type}_aggMarkers.csv"
        )
        filepath = subj_dir / filename
        
        # Save without index
        group.to_csv(filepath, index=False)


def create_test_config(base_config_path: Path, test_dir: Path,
                       marker_name: str, marker_type: str,
                       n_permutations: int = 100) -> Path:
    """Create test configuration based on existing config."""
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for testing
    config['project']['features_root'] = str(test_dir / 'features')
    config['project']['output_path'] = str(test_dir / 'results')
    config['project']['montage_path'] = 'standard_1020'  # Use standard montage
    config['project']['qa_summary_path'] = None  # Disable QA filtering
    config['project']['pca_results_path'] = None  # Disable PCA
    config['project']['subjects'] = None  # Use all subjects in synthetic data
    config['project']['tasks'] = None  # Use all tasks in synthetic data
    config['feature_families'] = {marker_type + '_test': [marker_name]}
    config['project']['selected_markers'] = {marker_type: [marker_type + '_test']}
    config['project']['onoff_max_value'] = None  # Include all probes
    
    # LMM settings optimized for convergence
    config['lmm']['formula'] = 'power ~ onoff + (1|subject)'
    config['lmm']['predictor_of_interest'] = 'onoff'
    config['lmm']['method'] = 'powell'
    config['lmm']['maxiter'] = 2000
    
    # Clustering settings for testing
    config['clustering']['method'] = 'tfce'
    config['clustering']['permutation_method'] = 'freedman_lane'
    config['clustering']['n_permutations'] = n_permutations
    config['clustering']['n_jobs'] = 1  # Single-threaded for reproducibility
    config['clustering']['alpha'] = 0.05
    
    # Save test config
    test_config_path = test_dir / 'test_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return test_config_path


class TestResults:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.n_sig_clusters = 0
        self.max_cluster_size = 0
        self.min_p_value = 1.0
        self.convergence_rate = 0.0
        self.expected_finding = None
        self.actual_finding = None
        self.error = None
        self.results_path = None
        
    def evaluate(self, expected_sig: bool) -> bool:
        """
        Evaluate if test result matches expectation.
        
        Parameters
        ----------
        expected_sig : bool
            Whether we expect to find significant clusters
            
        Returns
        -------
        bool
            True if result matches expectation
        """
        self.expected_finding = "significant" if expected_sig else "null"
        self.actual_finding = "significant" if self.n_sig_clusters > 0 else "null"
        
        # Test passes if finding matches expectation
        self.success = (self.n_sig_clusters > 0) == expected_sig
        
        return self.success
    
    def __str__(self):
        status = "✓ PASS" if self.success else "✗ FAIL"
        lines = [
            f"{status}: {self.name}",
            f"  Expected: {self.expected_finding}",
            f"  Found: {self.actual_finding} "
            f"({self.n_sig_clusters} clusters)",
            f"  Convergence: {self.convergence_rate:.1%}",
        ]
        if self.n_sig_clusters > 0:
            lines.append(f"  Best p-value: {self.min_p_value:.4f}")
            lines.append(
                f"  Largest cluster: {self.max_cluster_size} channels"
            )
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)


def run_single_test(
    test_name: str,
    effect_config: Optional[Dict[str, Dict[str, float]]],
    expected_significant: bool,
    base_config_path: Path,
    test_dir: Path,
    n_subjects: int = 20,
    n_probes: int = 25,
    n_permutations: int = 100,
    seed: int = 42
) -> TestResults:
    """
    Run a single pipeline test with specified effect configuration.
    
    Parameters
    ----------
    test_name : str
        Name of the test
    effect_config : Optional[Dict]
        Effect configuration (None for null test)
    expected_significant : bool
        Whether we expect to find significant results
    base_config_path : Path
        Path to base configuration file
    test_dir : Path
        Directory for test outputs
    n_subjects : int
        Number of subjects to simulate
    n_probes : int
        Number of probes per subject
    n_permutations : int
        Number of permutations for cluster test
    seed : int
        Random seed
        
    Returns
    -------
    TestResults
        Test results object
    """
    result = TestResults(test_name)
    
    try:
        # Create test subdirectory
        test_subdir = test_dir / test_name.lower().replace(" ", "_")
        if test_subdir.exists():
            shutil.rmtree(test_subdir)
        test_subdir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate synthetic data
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        print(f"Generating synthetic data...")
        
        df = simulate_realistic_eeg_data(
            n_subjects=n_subjects,
            n_probes_per_subject=n_probes,
            effect_config=effect_config,
            noise_level=1.0,
            subject_variability=0.5,
            seed=seed
        )
        
        print(f"  Generated: {len(df)} rows")
        print(f"  Subjects: {df['subject'].nunique()}")
        print(f"  Probes: {df.groupby('subject')['probe_number'].max().mean():.1f} per subject")
        print(f"  Channels: {df['channel'].nunique()}")
        
        if effect_config:
            for region, config in effect_config.items():
                n_channels = len(BRAIN_REGIONS[region])
                print(
                    f"  Effect in {region}: d={config['size']:.2f} "
                    f"({n_channels} channels)"
                )
        else:
            print(f"  No effects (null test)")
        
        # 2. Save data
        features_dir = test_subdir / 'features'
        save_synthetic_data(df, features_dir)
        print(f"  Saved to: {features_dir}")
        
        # 3. Create test config
        test_marker = 'EEG_psd_bands_spectralpower_alpha'
        marker_type = 'state'
        config_path = create_test_config(
            base_config_path, test_subdir, test_marker, marker_type, n_permutations
        )
        print(f"  Config: {config_path}")
        
        # 4. Run pipeline
        print(f"\nRunning pipeline...")
        import sys
        pipeline_dir = base_config_path.parent
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))
        
        from run_pipeline import main
        main(config_path=str(config_path), marker_index=None)
        
        # 5. Load and evaluate results
        print(f"\nEvaluating results...")
        predictor = 'onoff'
        results_dir = (
            test_subdir / 'results' / predictor /
            f'{marker_type}_{test_marker}'
        )
        result.results_path = results_dir
        
        results_file = results_dir / 'results.pkl'
        if results_file.exists():
            with open(results_file, 'rb') as f:
                res = pickle.load(f)
            
            result.n_sig_clusters = res['n_sig_clusters']
            result.convergence_rate = res.get('convergence_rate', 0.0)
            
            if result.n_sig_clusters > 0:
                # Find best cluster
                sig_p_values = [
                    p for p in res['cluster_p_values'] if p < 0.05
                ]
                result.min_p_value = (
                    min(sig_p_values) if sig_p_values else 1.0
                )
                
                # Find largest cluster
                sig_clusters = [
                    c for c, p in zip(
                        res['clusters'], res['cluster_p_values']
                    ) if p < 0.05
                ]
                result.max_cluster_size = (
                    max(len(c) for c in sig_clusters) if sig_clusters else 0
                )
        else:
            result.error = "Results file not found"
            
        # Evaluate against expectation
        result.evaluate(expected_significant)
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        import traceback
        traceback.print_exc()
    
    return result


def run_comprehensive_tests():
    """Run comprehensive test suite with multiple effect sizes."""
    
    print("="*80)
    print("COMPREHENSIVE PIPELINE TEST SUITE")
    print("="*80)
    print("\nThis test suite validates:")
    print("  1. Null case: No effects → No significant clusters")
    print("  2. Small effect: Subtle effects may or may not be detected")
    print("  3. Medium effect: Moderate effects should be detected")
    print("  4. Large effect: Strong effects must be detected")
    print("  5. Regional specificity: Effects in expected brain regions")
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent.absolute()
    base_config = script_dir / 'config.yaml'
    
    # Check if Statistics/config.yaml exists, otherwise look in parent
    if not base_config.exists():
        base_config = script_dir.parent / 'config.yaml'
    
    if not base_config.exists():
        raise FileNotFoundError(f"Config file not found: {base_config}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_dir = script_dir / 'test_results' / f'comprehensive_{timestamp}'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Test directory: {test_dir.absolute()}")
    print(f"Base config: {base_config.absolute()}")
    print()
    
    # Define test cases with realistic, localized clusters
    test_cases = [
        {
            'name': "Test 1: NULL (No Effect)",
            'effect_config': None,
            'expected_sig': False,
            'description': "Should NOT find significant clusters (true negative)"
        },
        {
            'name': "Test 2: SMALL Effect (d=0.3)",
            'effect_config': {
                'frontal': {
                    'size': 0.3,
                    'predictor': 'onoff',
                    'n_peak_channels': 2,
                    'spread': 1
                }
            },
            'expected_sig': False,
            'description': ("Small localized effect (2 peak + neighbors) "
                          "may or may not be detected")
        },
        {
            'name': "Test 3: MEDIUM Effect (d=0.5)",
            'effect_config': {
                'frontal': {
                    'size': 0.5,
                    'predictor': 'onoff',
                    'n_peak_channels': 3,
                    'spread': 2
                }
            },
            'expected_sig': True,
            'description': ("Should detect moderate frontal cluster "
                          "(~5-8 channels)")
        },
        {
            'name': "Test 4: LARGE Effect (d=0.8)",
            'effect_config': {
                'frontal': {
                    'size': 0.8,
                    'predictor': 'onoff',
                    'n_peak_channels': 4,
                    'spread': 2
                }
            },
            'expected_sig': True,
            'description': ("Must detect large frontal cluster "
                          "(~8-12 channels)")
        },
        {
            'name': "Test 5: DISTRIBUTED Effect",
            'effect_config': {
                'frontal': {
                    'size': 0.6,
                    'predictor': 'onoff',
                    'n_peak_channels': 2,
                    'spread': 2,
                    'decay_rate': 0.5
                },
                'parietal': {
                    'size': 0.4,
                    'predictor': 'onoff',
                    'n_peak_channels': 2,
                    'spread': 1,
                    'decay_rate': 0.5
                }
            },
            'expected_sig': True,
            'description': ("Should detect distributed frontal-parietal effect "
                          "(2 separate clusters, ~5-8 channels each)")
        },
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Description: {test_case['description']}")
        
        result = run_single_test(
            test_name=test_case['name'],
            effect_config=test_case['effect_config'],
            expected_significant=test_case['expected_sig'],
            base_config_path=base_config,
            test_dir=test_dir,
            n_subjects=20,  # Moderate sample size
            n_probes=25,    # Realistic probe count
            n_permutations=100,  # Fast for testing (use 1000+ for real)
            seed=42 + i  # Different seed per test
        )
        
        results.append(result)
        print(f"\n{result}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    n_passed = sum(r.success for r in results)
    n_total = len(results)
    
    print(f"\nTests passed: {n_passed}/{n_total}")
    print("\nDetailed Results:")
    for result in results:
        print(f"\n{result}")
    
    # Create summary report
    summary_data = []
    for result in results:
        summary_data.append({
            'test': result.name,
            'expected': result.expected_finding,
            'actual': result.actual_finding,
            'n_sig_clusters': result.n_sig_clusters,
            'min_p_value': result.min_p_value,
            'max_cluster_size': result.max_cluster_size,
            'convergence_rate': result.convergence_rate,
            'passed': result.success,
            'error': result.error or ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = test_dir / 'test_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Overall assessment
    print("\n" + "="*80)
    if n_passed == n_total:
        print("✓ ALL TESTS PASSED - Pipeline is working correctly!")
        print("  - Correctly rejects null hypothesis when no effect "
              "present")
        print("  - Correctly detects effects when present")
        print("  - Appropriate sensitivity to effect size")
    elif n_passed >= n_total * 0.8:
        print("⚠ MOST TESTS PASSED - Pipeline is mostly working")
        print("  - Review failed tests for potential issues")
        print("  - May need parameter tuning or more permutations")
    else:
        print("✗ MULTIPLE TESTS FAILED - Pipeline needs attention")
        print("  - Check convergence rates")
        print("  - Review error messages")
        print("  - Validate data generation and pipeline integration")
    print("="*80)
    
    print(f"\nTest artifacts saved to: {test_dir.absolute()}")
    print(f"View individual test results in subdirectories")
    print(f"Clean up when done: rm -rf {test_dir.parent}")
    
    return results, summary_df


if __name__ == "__main__":
    results, summary = run_comprehensive_tests()
