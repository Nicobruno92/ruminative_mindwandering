"""
Test script for predictor variability filtering.

This script demonstrates how the variability filter works with synthetic data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reader import filter_subjects_by_variability


def create_test_data():
    """Create synthetic data with varying levels of subject variability."""
    np.random.seed(42)
    
    data = []
    
    # Subject 1: Zero variance (always onoff=50)
    for i in range(10):
        data.append({'subject': '01', 'onoff': 50, 'probe': i})
    
    # Subject 2: Very low variance (std ≈ 1)
    for i in range(10):
        data.append({'subject': '02', 'onoff': 50 + np.random.randn() * 1, 'probe': i})
    
    # Subject 3: Low variance (std ≈ 5)
    for i in range(10):
        data.append({'subject': '03', 'onoff': 50 + np.random.randn() * 5, 'probe': i})
    
    # Subject 4: Medium variance (std ≈ 10)
    for i in range(10):
        data.append({'subject': '04', 'onoff': 50 + np.random.randn() * 10, 'probe': i})
    
    # Subject 5: High variance (std ≈ 20)
    for i in range(10):
        data.append({'subject': '05', 'onoff': 50 + np.random.randn() * 20, 'probe': i})
    
    df = pd.DataFrame(data)
    return df


def test_auto_filter():
    """Test automatic filtering (removes zero variance only)."""
    print("="*60)
    print("TEST 1: Auto filter (removes zero variance)")
    print("="*60)
    
    df = create_test_data()
    
    # Show original stats
    print("\nOriginal subject statistics:")
    stats = df.groupby('subject')['onoff'].agg(['std', 'mean', 'count'])
    print(stats)
    
    # Apply auto filter
    df_filtered = filter_subjects_by_variability(
        df=df,
        predictor_column='onoff',
        min_variability='auto',
        verbose=True
    )
    
    print(f"\nObservations: {len(df)} -> {len(df_filtered)}")
    print(f"Subjects: {df['subject'].nunique()} -> {df_filtered['subject'].nunique()}")
    

def test_numeric_threshold():
    """Test numeric threshold filtering."""
    print("\n" + "="*60)
    print("TEST 2: Numeric threshold (std > 5)")
    print("="*60)
    
    df = create_test_data()
    
    # Apply numeric threshold
    df_filtered = filter_subjects_by_variability(
        df=df,
        predictor_column='onoff',
        min_variability=5,
        verbose=True
    )
    
    print(f"\nObservations: {len(df)} -> {len(df_filtered)}")
    print(f"Subjects: {df['subject'].nunique()} -> {df_filtered['subject'].nunique()}")


def test_quantile_filter():
    """Test quantile-based filtering."""
    print("\n" + "="*60)
    print("TEST 3: Quantile filter (bottom 40%)")
    print("="*60)
    
    df = create_test_data()
    
    # Apply quantile filter
    df_filtered = filter_subjects_by_variability(
        df=df,
        predictor_column='onoff',
        min_variability='quantile_40',
        verbose=True
    )
    
    print(f"\nObservations: {len(df)} -> {len(df_filtered)}")
    print(f"Subjects: {df['subject'].nunique()} -> {df_filtered['subject'].nunique()}")


def test_no_filter():
    """Test no filtering."""
    print("\n" + "="*60)
    print("TEST 4: No filter")
    print("="*60)
    
    df = create_test_data()
    
    # Apply no filter
    df_filtered = filter_subjects_by_variability(
        df=df,
        predictor_column='onoff',
        min_variability=None,
        verbose=True
    )
    
    print(f"\nObservations: {len(df)} -> {len(df_filtered)}")
    print(f"Subjects: {df['subject'].nunique()} -> {df_filtered['subject'].nunique()}")
    assert len(df) == len(df_filtered), "No filter should not change data"


if __name__ == '__main__':
    print("Testing Predictor Variability Filter\n")
    
    test_auto_filter()
    test_numeric_threshold()
    test_quantile_filter()
    test_no_filter()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
