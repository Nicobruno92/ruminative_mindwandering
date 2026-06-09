"""Tests for Behavior/Classification/utils.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

# Allow import from Behavior/Classification/
sys.path.insert(0, str(Path(__file__).parent.parent / "Behavior" / "Classification"))
from utils import load_probe_data, load_pca_data, load_objective_markers, merge_all_features, get_feature_cols

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "Behavior" / "Classification" / "config.yaml"


@pytest.fixture
def config():
    return yaml.safe_load(open(CONFIG_PATH))


def test_load_probe_data_shape(config):
    df = load_probe_data(config)
    assert df.shape[0] > 0
    assert "subject" in df.columns
    assert "group" in df.columns
    assert "onoff" in df.columns
    assert "inclusion_exclusion" in df.columns


def test_load_pca_data_has_pcs(config):
    df = load_pca_data(config)
    assert "PC1" in df.columns
    assert "PC2" in df.columns
    assert "PC3" in df.columns
    assert df.shape[0] > 0


def test_load_objective_markers_has_cols(config):
    df = load_objective_markers(config)
    for col in ["omission_rate", "commission_rate", "rt_mean", "rtcv"]:
        assert col in df.columns


def test_merge_all_features_no_duplicate_columns(config):
    df = merge_all_features(config)
    assert df.columns.duplicated().sum() == 0


def test_merge_all_features_pc_columns_present(config):
    df = merge_all_features(config)
    assert "PC1" in df.columns


def test_get_feature_cols_with_pca_and_markers(config):
    cols = get_feature_cols(config)
    assert "onoff" in cols
    assert "PC1" in cols
    assert "omission_rate" in cols
    assert len(cols) == 5 + 3 + 4  # probe dims + PCs + objective markers
