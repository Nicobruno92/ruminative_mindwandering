import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path to import modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lmm_connectivity import run_lmm_per_connection, apply_fdr_correction

@pytest.fixture
def mock_lmm_data():
    """Create mock wide-format connectivity data for testing LMM."""
    # 5 subjects, 4 probes each = 20 observations
    subjects = ["01", "02", "03", "04", "05"]
    n_probes_per_sub = 4
    
    data = []
    for sub in subjects:
        for p in range(n_probes_per_sub):
            data.append({
                "subject": sub,
                "task": "Sart1",
                "probe_number": p + 1,
                "onoff": np.random.uniform(0, 100),
                "Fp1--F3": np.random.normal(0.5, 0.1),
                "F3--C3": np.random.normal(0.3, 0.1),
                "C3--P3": np.random.normal(0.4, 0.1)
            })
            
    df = pd.DataFrame(data)
    
    # Induce an effect in Fp1--F3 based on onoff
    df["Fp1--F3"] += df["onoff"] * 0.05
    
    connection_ids = ["Fp1--F3", "F3--C3", "C3--P3"]
    return df, connection_ids

def test_run_lmm_per_connection(mock_lmm_data):
    df_wide, connection_ids = mock_lmm_data
    
    results = run_lmm_per_connection(
        df_wide, connection_ids,
        formula="power ~ onoff + (1|subject)",
        predictor_of_interest="onoff"
    )
    
    # Check shape and columns
    assert len(results) == len(connection_ids)
    assert "connection_id" in results.columns
    assert "t_statistic" in results.columns
    assert "p_value" in results.columns
    assert "converged" in results.columns
    
    # Fp1--F3 should have a stronger effect
    fp1_f3 = results[results["connection_id"] == "Fp1--F3"].iloc[0]
    assert not np.isnan(fp1_f3["t_statistic"])
    assert fp1_f3["converged"]

def test_apply_fdr_correction(mock_lmm_data):
    df_wide, connection_ids = mock_lmm_data
    
    results = run_lmm_per_connection(df_wide, connection_ids)
    corrected = apply_fdr_correction(results, alpha=0.05, method="fdr_bh")
    
    assert "p_value_fdr" in corrected.columns
    assert "significant_fdr" in corrected.columns
    
    # The FDR p-value should be >= the uncorrected p-value for the smallest p-value
    # (Because there are multiple tests)
    min_pval_idx = corrected["p_value"].idxmin()
    assert corrected.loc[min_pval_idx, "p_value_fdr"] >= corrected.loc[min_pval_idx, "p_value"]
