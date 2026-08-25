"""
Classification performance tests to verify the Within-Subject pipeline actually learns from data.

Covers:
- High separability data (signal) -> High AUC (> 0.8)
- Random noise data -> Random AUC (~ 0.5)
- Testing GroupKFold natively with 'tasks'
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.ml_utils import run_within_subject_cv

def test_within_subject_learns_good_data():
    """Verify that highly separable data yields a high AUC for a single subject."""
    rng = np.random.default_rng(42)
    
    n_samples = 100
    X_0 = rng.normal(loc=-2.0, scale=0.5, size=(n_samples // 2, 5))
    y_0 = np.zeros(n_samples // 2)
    
    X_1 = rng.normal(loc=2.0, scale=0.5, size=(n_samples // 2, 5))
    y_1 = np.ones(n_samples // 2)
    
    X = pd.DataFrame(np.vstack([X_0, X_1]), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.concatenate([y_0, y_1]))
    
    # Stratified KFold Strategy
    result = run_within_subject_cv(
        X, y,
        cv_strategy="stratified_kfold",
        cv_folds=5,
        model_type="lr",
        fixed_random_state=42
    )
    
    auc = result["mean_auc"].values[0]
    assert auc > 0.85, f"Within-Subject pipeline failed to learn highly separable data. got {auc}"

def test_within_subject_fails_on_random_noise():
    """Verify that pure random noise yields an AUC close to 0.5 (chance)."""
    rng = np.random.default_rng(42)
    
    n_samples = 100
    X = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(n_samples, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.choice([0, 1], size=n_samples))
    
    result = run_within_subject_cv(
        X, y,
        cv_strategy="stratified_kfold",
        cv_folds=5,
        model_type="lr",
        fixed_random_state=42
    )
    
    auc = result["mean_auc"].values[0]
    assert 0.35 < auc < 0.65, f"Pipeline found signal in random noise. Expected AUC ~0.5, got {auc}"

def test_within_subject_group_kfold():
    """Verify GroupKFold respects group assignments (e.g. task sessions)."""
    rng = np.random.default_rng(42)
    
    n_samples = 120
    X_0 = rng.normal(loc=-2.0, scale=0.5, size=(n_samples // 2, 5))
    y_0 = np.zeros(n_samples // 2)
    
    X_1 = rng.normal(loc=2.0, scale=0.5, size=(n_samples // 2, 5))
    y_1 = np.ones(n_samples // 2)
    
    X = pd.DataFrame(np.vstack([X_0, X_1]), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.concatenate([y_0, y_1]))
    
    # 4 groups (e.g., Sart1, Sart2, Sart3, Sart4)
    tasks_0 = np.repeat([f"Sart{i}" for i in range(1, 5)], 15)
    tasks_1 = np.repeat([f"Sart{i}" for i in range(1, 5)], 15)
    groups = pd.Series(np.concatenate([tasks_0, tasks_1]))
    
    result = run_within_subject_cv(
        X, y, groups,
        cv_strategy="group_kfold",
        cv_folds=4,
        model_type="lr",
        fixed_random_state=42
    )
    
    auc = result["mean_auc"].values[0]
    assert auc > 0.85, f"GroupKFold pipeline failed to learn. Expected high AUC, got {auc}"
    
    cv_splits = result["cv_splits"].values[0]
    assert len(cv_splits) == 4, f"Expected 4 splits for GroupKFold(4), got {len(cv_splits)}"
