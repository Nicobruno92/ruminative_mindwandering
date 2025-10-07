#!/usr/bin/env python3
"""
Simple test script to check if the ERP analysis works with the fixes
"""

import os
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, './')

from utils.analysis_helpers import compute_erps, fit_lmm_for_time_bins, fit_true_mixed_effects_model

# Settings
subjects = [f"{i:02}" for i in range(2, 43)]
conditions_of_interest = ['go/correct/ontask', 'go/correct/offtask']
posterior_roi = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']

# Simple time bins for testing
time_bins_test = [
    (-0.1, 0.0),   # Baseline
    (0.4, 0.8),    # P3 window
]

# Test with 'highlow' metric (usually has strongest effects)
metric = 'highlow'
evoked_file_path = f"./results/ERPs/participant_evokeds_{metric}_car.pkl"

print(f"=== TESTING WITH METRIC: {metric} ===")

# Load data
if not os.path.exists(evoked_file_path):
    print(f"ERROR: File {evoked_file_path} not found!")
    exit(1)

print(f"Loading data from {evoked_file_path}...")
with open(evoked_file_path, 'rb') as file:
    participant_evokeds = pickle.load(file)

# Check data quality
total_subjects = 0
total_trials = 0
for subj_id, conditions in participant_evokeds.items():
    has_data = False
    for condition in conditions_of_interest:
        if condition in conditions and conditions[condition]:
            has_data = True
            total_trials += len(conditions[condition])
    if has_data:
        total_subjects += 1

print(f"Data quality check:")
print(f"  - Subjects with data: {total_subjects}/{len(subjects)}")
print(f"  - Total trials: {total_trials}")

# Test ERP computation with 'subject' aggregation
print(f"\nTesting ERP computation with subject-level data...")
erp_data = compute_erps(
    participant_evokeds=participant_evokeds,
    subjects=subjects,
    conditions_of_interest=conditions_of_interest,
    roi=posterior_roi,
    time_bins=time_bins_test,
    aggregate='subject'  # This should preserve Subject column
)

print(f"ERP data shape: {erp_data.shape}")
print(f"Columns: {list(erp_data.columns)}")
print(f"Sample data:")
print(erp_data.head())

# Test mixed-effects model
if 'Subject' in erp_data.columns:
    print(f"\n✓ Subject column found! Testing mixed-effects model...")
    try:
        lmm_results = fit_true_mixed_effects_model(erp_data)
        print(f"✓ Mixed-effects model SUCCESS!")
        print(f"Results shape: {lmm_results.shape}")
        print(lmm_results)
    except Exception as e:
        print(f"✗ Mixed-effects model failed: {e}")
        print(f"Falling back to standard LMM...")
        lmm_results = fit_lmm_for_time_bins(erp_data)
        print(f"Standard LMM results:")
        print(lmm_results)
else:
    print(f"✗ No Subject column found!")
    print(f"Available columns: {list(erp_data.columns)}")

print(f"\n=== TEST COMPLETED ===") 