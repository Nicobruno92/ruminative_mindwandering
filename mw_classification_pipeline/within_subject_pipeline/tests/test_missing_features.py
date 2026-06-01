"""
Unit tests for handle_missing_features (utils/data_utils.py).

Regression guard for the bug where partially-NaN event-property markers
(spindles/slowwaves) silently erased the entire dataset because the pipeline
dropped every sample containing any residual NaN.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data_utils import handle_missing_features


def _make_sparse_matrix():
    """
    4 subjects × 5 probes = 20 samples, 4 features:
      - dense_a, dense_b : no NaN
      - sparse_evt       : 60% NaN globally (event-property marker)
      - mild             : one NaN per subject (12.5% globally)
    """
    rng = np.random.default_rng(0)
    subjects = np.repeat(["02", "03", "04", "05"], 5)
    n = len(subjects)
    X = pd.DataFrame({
        "dense_a": rng.normal(size=n),
        "dense_b": rng.normal(size=n),
        "sparse_evt": rng.normal(size=n),
        "mild": rng.normal(size=n),
    })
    # sparse_evt: 60% NaN
    X.loc[rng.choice(n, size=12, replace=False), "sparse_evt"] = np.nan
    # mild: exactly one NaN in each subject's block
    for i in range(4):
        X.loc[i * 5, "mild"] = np.nan
    groups = pd.Series(subjects)
    return X, groups


def test_does_not_erase_all_samples_with_imputation():
    """The core regression: sparse columns must not nuke every sample."""
    X, groups = _make_sparse_matrix()
    X_clean, valid_mask, feature_cols = handle_missing_features(
        X, groups, max_feature_nan_frac=0.25, imputation="median", verbose=False
    )
    # All 20 samples retained
    assert valid_mask.all()
    assert len(X_clean) == 20
    # No NaN left
    assert not X_clean.isna().any().any()


def test_drops_only_oversparse_columns():
    """Columns above the NaN threshold are dropped; others survive."""
    X, groups = _make_sparse_matrix()
    _, _, feature_cols = handle_missing_features(
        X, groups, max_feature_nan_frac=0.25, imputation="median", verbose=False
    )
    # sparse_evt (60% NaN) dropped; mild (12.5%) kept
    assert "sparse_evt" not in feature_cols
    assert "mild" in feature_cols
    assert "dense_a" in feature_cols and "dense_b" in feature_cols


def test_per_subject_median_fill_uses_subject_context():
    """A subject's NaN is filled with that subject's own median, not global."""
    subjects = np.repeat(["A", "B"], 4)
    X = pd.DataFrame({"f": [10.0, 10.0, 10.0, np.nan, 0.0, 0.0, 0.0, 0.0]})
    groups = pd.Series(subjects)
    X_clean, valid_mask, _ = handle_missing_features(
        X, groups, max_feature_nan_frac=0.5, imputation="median", verbose=False
    )
    assert valid_mask.all()
    # Subject A's NaN (row 3) -> A's median (10.0), not the global median (5.0)
    assert X_clean.loc[3, "f"] == 10.0


def test_imputation_none_drops_residual_nan_samples():
    """With imputation='none', residual-NaN samples are dropped (not all)."""
    X, groups = _make_sparse_matrix()
    X_clean, valid_mask, feature_cols = handle_missing_features(
        X, groups, max_feature_nan_frac=0.25, imputation="none", verbose=False
    )
    # sparse_evt dropped by threshold; 'mild' still has 4 NaN -> 4 samples dropped
    assert "sparse_evt" not in feature_cols
    assert (~valid_mask).sum() == 4
    assert len(X_clean) == 16
    assert not X_clean.isna().any().any()


def test_threshold_zero_keeps_only_fully_observed():
    """max_feature_nan_frac=0 keeps only columns with no NaN at all."""
    X, groups = _make_sparse_matrix()
    _, valid_mask, feature_cols = handle_missing_features(
        X, groups, max_feature_nan_frac=0.0, imputation="median", verbose=False
    )
    assert set(feature_cols) == {"dense_a", "dense_b"}
    assert valid_mask.all()


def test_invalid_imputation_raises():
    X, groups = _make_sparse_matrix()
    with pytest.raises(ValueError, match="imputation must be"):
        handle_missing_features(X, groups, imputation="bogus", verbose=False)
