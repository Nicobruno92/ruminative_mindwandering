# LMM Cluster Permutation Test Suite

Comprehensive test suite for the Linear Mixed Model (LMM) based spatial cluster permutation testing pipeline, following MNE-Python testing patterns.

## Overview

This test suite validates the custom LMM cluster permutation implementation that adapts MNE's cluster permutation approach for mixed-effects models. The tests ensure:

1. **Statistical correctness**: Proper null distributions, effect detection, false positive control
2. **MNE compatibility**: Integration with MNE's adjacency structures and spatial operations  
3. **Robustness**: Proper error handling and edge case management
4. **Reproducibility**: Consistent results with fixed random seeds

## Test Structure

### 1. LMM Model Tests (`test_lmm_*.py`)

Tests the core Linear Mixed Model fitting per channel:

- **`test_parse_random_effects()`**: Validates parsing of R-style LMM formulas
- **`test_lmm_per_channel_basic()`**: Basic LMM fitting produces valid statistics
- **`test_lmm_detects_effect()`**: LMM correctly detects planted effects
- **`test_lmm_null_distribution()`**: Null distribution is approximately normal

**Why this matters**: LMM replaces simple t-tests to account for within-subject correlation and random effects.

### 2. Cluster Finding Tests (`test_find_clusters_*.py`)

Tests spatial cluster identification using MNE's core functions:

- **`test_validate_adjacency_matrix()`**: Adjacency validation (symmetric, correct shape)
- **`test_find_clusters_basic()`**: Basic cluster finding with linear adjacency
- **`test_find_clusters_no_adjacency()`**: Cluster finding without spatial structure
- **`test_find_clusters_tails()`**: One-tailed vs two-tailed cluster detection
- **`test_compute_cluster_statistic()`**: Cluster statistics with different t_power

**Why this matters**: Uses MNE's optimized `_get_components()` for spatial clustering.

### 3. Permutation Tests (`test_permute_*.py`, `test_cluster_permutation_*.py`)

Tests the permutation logic that preserves within-subject structure:

- **`test_permute_within_subjects()`**: Permutation preserves subject blocking
- **`test_prepare_permutation_data()`**: Data preparation for permutation
- **`test_cluster_permutation_basic()`**: Basic cluster permutation pipeline
- **`test_cluster_permutation_detects_effect()`**: Power to detect planted effects
- **`test_cluster_permutation_null()`**: False positive control under null
- **`test_cluster_permutation_tails()`**: Different tail settings work correctly
- **`test_permutation_reproducibility()`**: Results are reproducible with same seed

**Why this matters**: Custom permutation logic is needed to maintain LMM's within-subject structure.

### 4. TFCE Tests (`test_tfce_*.py`)

Tests Threshold-Free Cluster Enhancement:

- **`test_tfce_basic()`**: Basic TFCE pipeline works
- **`test_tfce_threshold_parameters()`**: Threshold parameters validated correctly

**Why this matters**: TFCE provides threshold-free approach that can be more sensitive.

### 5. Integration Tests

Tests integration with real MNE components:

- **`test_cluster_permutation_realistic_montage()`**: Works with MNE channel montages
- **`test_cluster_permutation_invalid_inputs()`**: Proper error handling
- **`test_lmm_convergence_warnings()`**: Handles convergence issues gracefully

## Running the Tests

### Run all tests:
```bash
cd Statistics/tests
pytest test_lmm_cluster_permutation.py -v
```

### Run specific test categories:
```bash
# LMM model tests only
pytest test_lmm_cluster_permutation.py -k "lmm" -v

# Cluster finding tests only  
pytest test_lmm_cluster_permutation.py -k "cluster" -v

# Permutation tests only
pytest test_lmm_cluster_permutation.py -k "permute" -v

# TFCE tests only
pytest test_lmm_cluster_permutation.py -k "tfce" -v
```

### Run with the convenience script:
```bash
cd Statistics/tests
./run_lmm_tests.sh
```

## Test Data

Tests use synthetic EEG data that mimics real experimental structure:

- **Simple EEG data**: Small dataset (10 subjects, 20 trials, 8 channels) with no effect
- **EEG data with effect**: Medium dataset (15 subjects, 30 trials, 10 channels) with planted effect (Cohen's d ≈ 0.6) in channels 2-4
- **Adjacency matrices**: 
  - Simple linear chain for basic tests
  - Realistic 10-20 montage from MNE for integration tests

## Expected Behavior

### Under Null Hypothesis:
- t-statistics follow approximately normal distribution (mean ≈ 0, std ≈ 1)
- False positive rate ≈ 5% for α = 0.05
- No significant clusters found in most runs

### With Real Effect:
- Higher t-statistics in effect channels
- Significant clusters detected (p < 0.05)
- Cluster locations match planted effect regions

### Reproducibility:
- Same random seed → identical results
- Different seeds → different permutation orders but similar statistical conclusions

## Comparison to MNE Tests

These tests follow patterns from MNE-Python's test suite:

| MNE Test | Our Equivalent | Purpose |
|----------|----------------|---------|
| `test_regression.py::test_regression()` | `test_lmm_per_channel_basic()` | Basic model fitting |
| `test_permutations.py::test_permutation_t_test()` | `test_cluster_permutation_basic()` | Permutation logic |
| `test_cluster_level.py::test_cluster_permutation_test()` | `test_cluster_permutation_detects_effect()` | Power/detection |
| `test_cluster_level.py::test_thresholds()` | `test_cluster_permutation_null()` | False positive control |
| `test_cluster_level.py::test_tfce_thresholds()` | `test_tfce_basic()` | TFCE functionality |
| `test_adjacency.py::test_adjacency_equiv()` | `test_validate_adjacency_matrix()` | Adjacency handling |

## Key Differences from MNE

1. **LMM instead of t-test**: We fit mixed models per channel, not simple t-tests
2. **Custom permutation**: Permutation preserves within-subject structure for LMM
3. **Integration**: Uses MNE's core cluster finding (`_get_components()`) but custom permutation logic

## Troubleshooting

### Test Failures

**`test_lmm_detects_effect` fails**:
- Check that effect size is sufficient (Cohen's d ≈ 0.6)
- Increase n_permutations for more stable p-values
- This test can occasionally fail due to randomness

**`test_cluster_permutation_null` fails**:
- False positive rate has natural variation
- Threshold allows up to 3 false positives in 20 runs (≈15% vs expected 5%)
- Consider running more replications

**Convergence warnings**:
- Normal for some channels with difficult data
- Tests should handle gracefully with NaN/Inf
- Check `test_lmm_convergence_warnings()`

### Dependencies

Required packages:
```bash
pip install pytest numpy pandas scipy statsmodels mne scikit-learn
```

Or use the eeg environment:
```bash
conda activate eeg
```

## References

1. **Maris & Oostenveld (2007)**: Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*, 164(1), 177-190.

2. **MNE-Python**: https://mne.tools/
   - `mne.stats.permutation_cluster_test`
   - `mne.stats.cluster_level._get_components`

3. **Freedman & Lane (1983)**: A nonstochastic interpretation of reported significance levels. *Journal of Business & Economic Statistics*, 1(4), 292-298.

## Contact

For questions or issues with the test suite, check:
- Main pipeline: `Statistics/README.md`
- Cluster testing: `Statistics/cluster_test.py`
- LMM implementation: `Statistics/lmm_model.py`
