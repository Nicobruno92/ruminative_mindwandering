"""Tests for scripts/plot_auc_vs_median_position.py. Synthetic data + real LOSO fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_auc_vs_median_position import (
    LOSO_RESULTS_ROOT,
    compute_pearson_fit,
    compute_subject_medians,
    build_fig_a_row,
    build_fig_b_rows,
)


def test_compute_pearson_fit_perfect_positive_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = compute_pearson_fit(x, y)
    assert result["r"] == pytest.approx(1.0)
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert result["n"] == 5


def test_compute_pearson_fit_returns_nan_below_three_points():
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    result = compute_pearson_fit(x, y)
    assert np.isnan(result["r"])
    assert np.isnan(result["slope"])
    assert result["n"] == 2


def test_compute_subject_medians_filters_and_computes_median():
    probe_df = pd.DataFrame({
        "subject": ["02", "02", "02", "03", "03", "04", "04"],
        "onoff": [10.0, 50.0, 90.0, 40.0, 60.0, 0.0, 100.0],
    })
    result = compute_subject_medians(probe_df, "onoff", subjects_final=["02", "03"])
    result = result.set_index("subject")
    assert result.loc["02", "subject_median"] == pytest.approx(50.0)
    assert result.loc["03", "subject_median"] == pytest.approx(50.0)
    assert "04" not in result.index


def test_build_fig_a_row_computes_sd_and_mean():
    subject_medians = pd.DataFrame({
        "subject": ["02", "03", "04"],
        "subject_median": [40.0, 50.0, 60.0],
    })
    subject_aucs = pd.DataFrame({
        "subject": ["02", "03", "04"],
        "auc": [0.6, 0.7, 0.8],
    })
    row = build_fig_a_row("Test Dim", subject_medians, subject_aucs)
    assert row["dimension"] == "Test Dim"
    assert row["median_sd"] == pytest.approx(10.0)
    assert row["mean_auc"] == pytest.approx(0.7)
    assert row["n_subjects"] == 3


def test_build_fig_b_rows_computes_distance_and_merges():
    subject_medians = pd.DataFrame({
        "subject": ["02", "03"],
        "subject_median": [50.0, 80.0],
    })
    subject_aucs = pd.DataFrame({
        "subject": ["02", "03"],
        "auc": [0.65, 0.55],
    })
    result = build_fig_b_rows("Test Dim", subject_medians, subject_aucs).set_index("subject")
    assert result.loc["02", "dist_from_50"] == pytest.approx(0.0)
    assert result.loc["03", "dist_from_50"] == pytest.approx(30.0)
    assert (result["dimension"] == "Test Dim").all()
    assert list(result.columns) == ["dimension", "subject_median", "dist_from_50", "auc"]


# =============================================================================
# Integration tests against real ON_vs_OFF LOSO results
# =============================================================================

from scripts.plot_auc_vs_median_position import load_dimension_metadata, load_subject_aucs

ON_OFF_RF_DIR = LOSO_RESULTS_ROOT / "ON_vs_OFF_within_median" / "all" / "rf"


def test_load_dimension_metadata_on_off():
    subjects_final, label_source = load_dimension_metadata(ON_OFF_RF_DIR)
    assert label_source == "onoff"
    assert len(subjects_final) == 29
    assert "03" in subjects_final
    assert "16" not in subjects_final
    assert all(len(s) == 2 for s in subjects_final)


def test_load_subject_aucs_on_off():
    subjects_final, _ = load_dimension_metadata(ON_OFF_RF_DIR)
    aucs = load_subject_aucs(ON_OFF_RF_DIR, subjects_final)
    assert len(aucs) == 29
    assert set(aucs["subject"]) == set(subjects_final)
    assert aucs["auc"].mean() == pytest.approx(0.628, abs=0.001)
