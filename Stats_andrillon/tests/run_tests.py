#!/usr/bin/env python
"""
Run all tests for Andrillon 2020 pipeline.

Usage:
    python run_tests.py
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))

from test_permutation import run_all_tests as run_permutation_tests
from test_clustering import run_all_tests as run_clustering_tests


def main():
    """Run all test suites."""
    print("\n" + "="*80)
    print("ANDRILLON 2020 PIPELINE - TEST SUITE")
    print("="*80)
    
    all_passed = True
    
    # Run permutation tests
    print("\n[1/2] Running permutation tests...")
    if not run_permutation_tests():
        all_passed = False
    
    # Run clustering tests
    print("\n[2/2] Running clustering tests...")
    if not run_clustering_tests():
        all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - PIPELINE READY")
        print("="*80)
        print("\nNext steps:")
        print("1. Test with real data: python run_andrillon_pipeline.py --config config_andrillon.yaml --marker <MARKER>")
        print("2. Run on cluster: sbatch submit_andrillon_array.sh")
        print("\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED - PLEASE FIX BEFORE PROCEEDING")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
