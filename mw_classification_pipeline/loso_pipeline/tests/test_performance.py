"""
Classification performance tests to verify the pipeline actually learns from data.

Covers:
- High separability data (signal) -> High AUC (> 0.8)
- Random noise data -> Random AUC (~ 0.5)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.ml_utils import run_model_pipeline_cv

def test_pipeline_learns_good_data():
    """Verify that highly separable data yields a high AUC."""
    # Generate directly separable data
    rng = np.random.default_rng(42)
    
    n_samples = 100
    # Group 1 (Class 0)
    X_0 = rng.normal(loc=-2.0, scale=0.5, size=(n_samples // 2, 5))
    y_0 = np.zeros(n_samples // 2)
    
    # Group 2 (Class 1)
    X_1 = rng.normal(loc=2.0, scale=0.5, size=(n_samples // 2, 5))
    y_1 = np.ones(n_samples // 2)
    
    X = pd.DataFrame(np.vstack([X_0, X_1]), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.concatenate([y_0, y_1]))
    
    # Fake 5 subjects, ensuring each subject gets 10 samples of class 0 and 10 of class 1
    subjects_0 = np.repeat([f"sub-{i:02d}" for i in range(1, 6)], (n_samples // 2) // 5)
    subjects_1 = np.repeat([f"sub-{i:02d}" for i in range(1, 6)], (n_samples // 2) // 5)
    groups = pd.Series(np.concatenate([subjects_0, subjects_1]))
    
    result = run_model_pipeline_cv(
        X, y, groups,
        model_type="lr",
        k=5,
        fixed_random_state=42
    )
    
    auc = result["mean_auc"].values[0]
    # Expected AUC should be very close to 1.0 due to extreme separability
    assert auc > 0.85, f"Pipeline failed to learn highly separable data. Expected high AUC, got {auc}"

def test_pipeline_fails_on_random_noise():
    """Verify that pure random noise yields an AUC close to 0.5 (chance)."""
    rng = np.random.default_rng(42)
    
    n_samples = 100
    # Both classes have the exact same distribution
    X = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(n_samples, 5)), columns=[f"f{i}" for i in range(5)])
    
    # Randomly assign 0 and 1
    y = pd.Series(rng.choice([0, 1], size=n_samples))
    
    # Fake 5 subjects
    subjects = np.repeat([f"sub-{i:02d}" for i in range(1, 6)], n_samples // 5)
    groups = pd.Series(subjects)
    
    result = run_model_pipeline_cv(
        X, y, groups,
        model_type="lr",
        k=5,
        fixed_random_state=42
    )
    
    auc = result["mean_auc"].values[0]
    # Expected AUC should be around 0.5 (chance level)
    # Give a bit of margin for random variation in small sample size
    assert 0.35 < auc < 0.65, f"Pipeline found signal in random noise. Expected AUC ~0.5, got {auc}"
