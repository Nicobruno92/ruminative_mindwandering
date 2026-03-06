import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reader import aggregate_to_roi_pairs, get_roi_pair_labels, _extract_band_from_column

def test_extract_band():
    assert _extract_band_from_column("wsmi_theta_pairs_trimmean") == "theta"
    assert _extract_band_from_column("wsmi_alpha_pairs_trimmean") == "alpha"
    assert _extract_band_from_column("other_col") == "unknown"


@pytest.fixture
def mock_connectivity_data():
    """Create mock long-format connectivity data."""
    data = []
    
    # 2 probes, 3 channel pairs
    for p in [1, 2]:
        for ch_pair, val in [("Fp1--F3", 0.5), ("Fp1--Fp2", 0.6), ("C3--P3", 0.4)]:
            data.append({
                "subject": "01",
                "task": "Sart1",
                "probe_number": p,
                "label": "MW",
                "n_epochs": 10,
                "onoff": 50,
                "epoch_type": "state_connectivity",
                "band": "theta",
                "connection_id": ch_pair,
                "wsmi_value": val + (p * 0.1) # Add variance per probe
            })
            
    return pd.DataFrame(data)


def test_aggregate_to_roi_pairs(mock_connectivity_data):
    rois = {
        "frontal_left": ["Fp1", "F3"],
        "frontal_right": ["Fp2", "F4"],
        "central_left": ["C3"],
        "posterior_left": ["P3"]
    }
    
    df = aggregate_to_roi_pairs(mock_connectivity_data, rois, verbose=False)
    
    # Expected ROI mapping:
    # Fp1--F3 -> frontal_left--frontal_left
    # Fp1--Fp2 -> frontal_left--frontal_right
    # C3--P3 -> central_left--posterior_left
    
    assert len(df) > 0
    assert "roi1" not in df.columns # internal columns should be dropped/re-grouped
    
    # Check that pairs are correctly named and sorted alphabetically
    pairs = df["connection_id"].unique()
    assert "frontal_left--frontal_left" in pairs
    assert "frontal_left--frontal_right" in pairs # Fp1--Fp2
    assert "central_left--posterior_left" in pairs # C3--P3
    
    # Check aggregation (average values)
    # Probe 1: Fp1--F3 = 0.6. Aggregated = 0.6
    p1_fl_fl = df[(df["probe_number"] == 1) & (df["connection_id"] == "frontal_left--frontal_left")]
    assert len(p1_fl_fl) == 1
    assert pytest.approx(p1_fl_fl.iloc[0]["wsmi_value"]) == 0.6

def test_get_roi_pair_labels():
    rois = {
        "A": ["1"],
        "B": ["2"],
    }
    
    pairs = get_roi_pair_labels(rois)
    assert len(pairs) == 3 # A-A, A-B, B-B
    assert "A--A" in pairs
    assert "A--B" in pairs
    assert "B--B" in pairs
    assert "B--A" not in pairs # Sorted alphabetically
