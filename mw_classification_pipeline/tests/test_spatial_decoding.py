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
