import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lmm_connectivity import run_lmm_per_connection, apply_fdr_correction

def generate_synthetic_data(
    n_subjects: int = 20,
    n_probes_per_sub: int = 20,
    effect_size: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Generate synthetic data with known ground truth effects.
    
    Creates 3 connections:
    - 'conn_null': Pure noise, no relation to onoff
    - 'conn_positive': Positive correlation with onoff
    - 'conn_negative': Negative correlation with onoff
    """
    rng = np.random.RandomState(random_state)
    
    data = []
    for sub_id in range(1, n_subjects + 1):
        # Subject-specific random intercepts (baseline offset)
        subj_intercept_null = rng.normal(0, 1.0)
        subj_intercept_pos = rng.normal(0, 1.0)
        subj_intercept_neg = rng.normal(0, 1.0)
        
        for p in range(n_probes_per_sub):
            # Predictor variable: mind-wandering continuous 0-100
            onoff = rng.uniform(0, 100)
            
            # Scale onoff to roughly 0-1 for effect generation
            onoff_scaled = onoff / 100.0
            
            # Ground truth generation:
            # y = baseline + intercept + (effect * onoff) + noise
            
            # 1. Null effect (pure noise + subject intercept)
            val_null = 0.5 + subj_intercept_null + rng.normal(0, 0.5)
            
            # 2. Positive effect
            val_pos = 0.5 + subj_intercept_pos + (effect_size * onoff_scaled) + rng.normal(0, 0.5)
            
            # 3. Negative effect
            val_neg = 0.5 + subj_intercept_neg - (effect_size * onoff_scaled) + rng.normal(0, 0.5)
            
            data.append({
                "subject": f"{sub_id:02d}",
                "task": "Sart1",
                "probe_number": p + 1,
                "onoff": onoff,
                "conn_null": val_null,
                "conn_positive": val_pos,
                "conn_negative": val_neg
            })
            
    df = pd.DataFrame(data)
    connection_ids = ["conn_null", "conn_positive", "conn_negative"]
    
    return df, connection_ids

def test_lmm_statistical_correctness():
    """
    Verify that the LMM correctly identifies true positive and true negative effects.
    """
    # Generate data with a strong effect size to ensure high statistical power
    df_wide, connection_ids = generate_synthetic_data(
        n_subjects=30, 
        n_probes_per_sub=15, 
        effect_size=3.0, # Strong effect
        random_state=123
    )
    
    results = run_lmm_per_connection(
        df_wide, 
        connection_ids,
        formula="power ~ onoff + (1|subject)",
        predictor_of_interest="onoff"
    )
    
    # 1. Check convergence
    assert results["converged"].all(), "All models should converge on clean synthetic data"
    
    # 2. Extract results for each connection
    res_null = results[results["connection_id"] == "conn_null"].iloc[0]
    res_pos = results[results["connection_id"] == "conn_positive"].iloc[0]
    res_neg = results[results["connection_id"] == "conn_negative"].iloc[0]
    
    # 3. STATISTICAL CHECKS
    
    # Null connection should NOT be significant (Type I error check)
    assert res_null["p_value"] > 0.05, f"False positive on robust null data (p={res_null['p_value']})"
    
    # Positive connection SHOULD be significant with positive t-stat (Type II error check)
    assert res_pos["p_value"] < 0.05, f"Failed to detect strong positive effect (p={res_pos['p_value']})"
    assert res_pos["t_statistic"] > 0, "Positive effect should yield positive t-statistic"
    assert res_pos["coefficient"] > 0, "Positive effect should yield positive coefficient"
    
    # Negative connection SHOULD be significant with negative t-stat (Type II error check)
    assert res_neg["p_value"] < 0.05, f"Failed to detect strong negative effect (p={res_neg['p_value']})"
    assert res_neg["t_statistic"] < 0, "Negative effect should yield negative t-statistic"
    assert res_neg["coefficient"] < 0, "Negative effect should yield negative coefficient"

def test_lmm_fdr_control():
    """
    Verify that FDR correction appropriately controls the false discovery rate
    when many null tests are performed alongside a few true effects.
    """
    # Create 97 null connections and 3 true positive connections
    n_nulls = 97
    df_wide, _ = generate_synthetic_data(n_subjects=20, n_probes_per_sub=15, effect_size=2.0)
    
    rng = np.random.RandomState(456)
    
    connection_ids = []
    
    # Add null noise connections
    for i in range(n_nulls):
        col_name = f"noise_{i}"
        df_wide[col_name] = rng.normal(0, 1, size=len(df_wide))
        connection_ids.append(col_name)
        
    # Add real effect connections
    for i in range(3):
        col_name = f"real_{i}"
        df_wide[col_name] = (df_wide["onoff"] / 100.0) * 1.5 + rng.normal(0, 0.5, size=len(df_wide))
        connection_ids.append(col_name)
        
    results = run_lmm_per_connection(
        df_wide, 
        connection_ids,
        formula="power ~ onoff + (1|subject)",
        predictor_of_interest="onoff"
    )
    
    # Apply FDR correction
    corrected = apply_fdr_correction(results, alpha=0.05, method="fdr_bh")
    
    # Check that FDR correction is more conservative than uncorrected
    # (The FDR p-value must be >= uncorrected p-value for all tests)
    assert (corrected["p_value_fdr"] >= corrected["p_value"] - 1e-10).all()
    
    # Check that real effects survived FDR (Statistical Power)
    real_results = corrected[corrected["connection_id"].str.startswith("real_")]
    assert real_results["significant_fdr"].all(), "True strong effects should survive FDR correction"
    
    # Check that null effects were correctly rejected (Type I error control)
    # At alpha 0.05, uncorrected might have ~5 false positives
    # FDR should ideally have 0-1 false positives among the 97 nulls here
    noise_results = corrected[corrected["connection_id"].str.startswith("noise_")]
    false_positives = noise_results["significant_fdr"].sum()
    
    # Assert strong control over false positives
    assert false_positives <= 2, f"FDR failed to control false positives (found {false_positives})"
