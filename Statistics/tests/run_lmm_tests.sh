#!/bin/bash
# =============================================================================
# Run LMM cluster permutation test suite
# =============================================================================

# Activate appropriate environment
# conda activate eeg  # Uncomment if needed

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "LMM Cluster Permutation Test Suite"
echo "======================================"
echo ""

# Run all tests with verbose output
echo "Running all tests..."
pytest test_lmm_cluster_permutation.py -v --tb=short

# Run specific test groups
echo ""
echo "======================================"
echo "Test Summary by Category"
echo "======================================"

echo ""
echo "1. LMM Model Tests:"
pytest test_lmm_cluster_permutation.py::test_parse_random_effects -v
pytest test_lmm_cluster_permutation.py::test_lmm_per_channel_basic -v
pytest test_lmm_cluster_permutation.py::test_lmm_detects_effect -v

echo ""
echo "2. Cluster Finding Tests:"
pytest test_lmm_cluster_permutation.py::test_validate_adjacency_matrix -v
pytest test_lmm_cluster_permutation.py::test_find_clusters_basic -v
pytest test_lmm_cluster_permutation.py::test_find_clusters_tails -v

echo ""
echo "3. Permutation Tests:"
pytest test_lmm_cluster_permutation.py::test_permute_within_subjects -v
pytest test_lmm_cluster_permutation.py::test_cluster_permutation_basic -v
pytest test_lmm_cluster_permutation.py::test_cluster_permutation_detects_effect -v
pytest test_lmm_cluster_permutation.py::test_permutation_reproducibility -v

echo ""
echo "4. TFCE Tests:"
pytest test_lmm_cluster_permutation.py::test_tfce_basic -v

echo ""
echo "======================================"
echo "Test Suite Complete"
echo "======================================"
