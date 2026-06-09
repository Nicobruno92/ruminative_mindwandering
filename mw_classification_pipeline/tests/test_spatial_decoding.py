"""Unit and smoke tests for utils/spatial_decoding_utils.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.spatial_decoding_utils import (
    parse_channels_from_columns,
    select_channel_columns,
)


def test_parse_channels_recovers_ordered_unique_set(per_channel_X):
    channels = parse_channels_from_columns(per_channel_X.columns.tolist())
    assert channels == ["Fz", "P1", "P10", "Pz"]  # sorted, unique


def test_parse_channels_does_not_confuse_P1_and_P10(per_channel_X):
    channels = parse_channels_from_columns(per_channel_X.columns.tolist())
    assert "P1" in channels and "P10" in channels


def test_select_channel_columns_returns_only_that_channel(per_channel_X):
    X_p1 = select_channel_columns(per_channel_X, "P1")
    assert all(c.startswith("P1_") for c in X_p1.columns)
    # P10 columns must NOT leak into P1
    assert not any(c.startswith("P10_") for c in X_p1.columns)
    assert X_p1.shape[1] == 3  # 3 markers


def test_select_channel_columns_raises_on_unknown_channel(per_channel_X):
    with pytest.raises(ValueError, match="no columns"):
        select_channel_columns(per_channel_X, "Cz")


from utils.spatial_decoding_utils import fdr_correct


def test_fdr_correct_matches_benjamini_hochberg_reference():
    # Known BH example: p = [0.01, 0.02, 0.03, 0.04, 0.05], n=5
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    p_adj, reject = fdr_correct(p, alpha=0.05)
    # BH adjusted: each p*(n/rank); monotone-enforced
    expected_adj = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(p_adj, expected_adj, rtol=1e-9)
    assert reject.tolist() == [True, True, True, True, True]


def test_fdr_correct_rejects_nothing_when_all_large():
    p = np.array([0.4, 0.6, 0.8])
    p_adj, reject = fdr_correct(p, alpha=0.05)
    assert reject.tolist() == [False, False, False]


from utils.spatial_decoding_utils import permutation_pvalue


def test_permutation_pvalue_plus_one_convention():
    # true=0.70, null has 9 values; 1 of them >= 0.70 -> p = (1+1)/(1+9) = 0.2
    null = [0.50, 0.55, 0.60, 0.45, 0.52, 0.58, 0.49, 0.71, 0.40]
    p = permutation_pvalue(true_value=0.70, null_values=null)
    assert p == pytest.approx((1 + 1) / (1 + 9))


def test_permutation_pvalue_floor_with_empty_null():
    # No null -> p = (1+0)/(1+0) = 1.0
    assert permutation_pvalue(0.7, []) == pytest.approx(1.0)
