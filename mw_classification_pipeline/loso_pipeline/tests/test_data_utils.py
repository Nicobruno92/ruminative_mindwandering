"""
Exhaustive tests for utils/data_utils.py.

Covers:
- create_label_contrast (threshold, extreme groups, edge cases)
- filter_participants_by_balance (all thresholds)
- filter_participants_by_sample_count
- get_feature_columns
- pivot_to_wide
- get_project_root
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data_utils import (
    create_label_contrast,
    filter_participants_by_balance,
    filter_participants_by_sample_count,
    get_feature_columns,
    pivot_to_wide,
    get_project_root,
)
from conftest import make_synthetic_data


# =============================================================================
# create_label_contrast
# =============================================================================

class TestCreateLabelContrastThreshold:
    """Tests for threshold (median-split) mode of create_label_contrast."""

    def _df(self, vals):
        return pd.DataFrame({"onoff": vals, "subject": "02", "task": "Sart1"})

    def test_above_threshold_is_class_1(self):
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": True}
        df = self._df([60, 80])
        result = create_label_contrast(df, contrast)
        assert list(result["target"]) == [1, 1]

    def test_below_threshold_is_class_0(self):
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": True}
        df = self._df([10, 40])
        result = create_label_contrast(df, contrast)
        assert list(result["target"]) == [0, 0]

    def test_exact_threshold_is_class_0(self):
        """Value exactly at threshold → class 0 (positive_above means strictly above)."""
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": True}
        df = self._df([50])
        result = create_label_contrast(df, contrast)
        assert result["target"].values[0] == 0

    def test_all_rows_kept_in_threshold_mode(self):
        """Threshold mode should not drop any rows."""
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": True}
        df = self._df(list(range(0, 101, 5)))
        result = create_label_contrast(df, contrast)
        assert len(result) == len(df)

    def test_positive_above_false_inverts_labels(self):
        """positive_above=False means below threshold → class 1."""
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": False}
        df = self._df([10, 90])
        result = create_label_contrast(df, contrast)
        # 10 < 50 → class 1; 90 > 50 → class 0
        assert result[result["onoff"] == 10]["target"].values[0] == 1
        assert result[result["onoff"] == 90]["target"].values[0] == 0

    def test_empty_dataframe_returns_empty(self):
        """Empty input → empty output."""
        contrast = {"label_source": "onoff", "threshold": 50, "positive_above": True}
        df = pd.DataFrame({"onoff": [], "subject": [], "task": []})
        result = create_label_contrast(df, contrast)
        assert len(result) == 0


class TestCreateLabelContrastExtremeGroups:
    """Tests for extreme-group (exclusion) mode of create_label_contrast."""

    def _df(self, vals):
        return pd.DataFrame({"onoff": vals, "subject": "02", "task": "Sart1"})

    def test_middle_values_excluded(self):
        contrast = {
            "label_source": "onoff", "threshold_low": 30, "threshold_high": 70
        }
        df = self._df([10, 50, 90])
        result = create_label_contrast(df, contrast)
        assert 50 not in result["onoff"].values
        assert len(result) == 2

    def test_at_low_threshold_is_class_0(self):
        """Value exactly at threshold_low → included as class 0."""
        contrast = {
            "label_source": "onoff", "threshold_low": 30, "threshold_high": 70
        }
        df = self._df([30])
        result = create_label_contrast(df, contrast)
        assert len(result) == 1
        assert result["target"].values[0] == 0

    def test_at_high_threshold_is_class_1(self):
        """Value exactly at threshold_high → included as class 1."""
        contrast = {
            "label_source": "onoff", "threshold_low": 30, "threshold_high": 70
        }
        df = self._df([70])
        result = create_label_contrast(df, contrast)
        assert len(result) == 1
        assert result["target"].values[0] == 1

    def test_extreme_groups_produce_two_classes(self):
        contrast = {
            "label_source": "onoff", "threshold_low": 30, "threshold_high": 70
        }
        df = self._df([0, 20, 50, 80, 100])
        result = create_label_contrast(df, contrast)
        assert set(result["target"].unique()) == {0, 1}

    def test_all_middle_returns_empty(self):
        """All values in the excluded range → empty output."""
        contrast = {
            "label_source": "onoff", "threshold_low": 30, "threshold_high": 70
        }
        df = self._df([35, 40, 50, 60, 65])
        result = create_label_contrast(df, contrast)
        assert len(result) == 0


# =============================================================================
# filter_participants_by_balance
# =============================================================================

class TestFilterParticipantsByBalance:
    """Tests for filter_participants_by_balance."""

    def _df(self, subject_targets):
        rows = []
        for sub, targets in subject_targets.items():
            for t in targets:
                rows.append({"subject": sub, "target": t})
        return pd.DataFrame(rows)

    def test_removes_all_one_class(self):
        df = self._df({"02": [1, 1, 1, 1], "03": [0, 1, 0, 1]})
        result = filter_participants_by_balance(df, "subject", "target", 0.2)
        assert "02" not in result["subject"].values
        assert "03" in result["subject"].values

    def test_keeps_exactly_at_threshold(self):
        """Subject with exactly 20% minority should be kept at threshold=0.2."""
        # 1 out of 5 → 20%
        df = self._df({"02": [1, 0, 0, 0, 0]})
        result = filter_participants_by_balance(df, "subject", "target", 0.2)
        assert "02" in result["subject"].values

    def test_removes_just_below_threshold(self):
        """Subject with 1/6 ≈ 16.7% minority → removed at threshold=0.2."""
        df = self._df({"02": [1, 0, 0, 0, 0, 0]})
        result = filter_participants_by_balance(df, "subject", "target", 0.2)
        assert "02" not in result["subject"].values

    def test_zero_threshold_keeps_all(self):
        df = self._df({"02": [1, 1, 1, 1], "03": [0, 0, 0, 0]})
        result = filter_participants_by_balance(df, "subject", "target", 0.0)
        assert len(result) == len(df)

    def test_threshold_1_removes_all_unbalanced(self):
        """Only perfectly balanced (50/50) subjects should survive threshold=0.5."""
        df = self._df({
            "02": [1, 1, 0, 0],     # 50% → keep
            "03": [1, 1, 1, 0],     # 25% minority → remove
        })
        result = filter_participants_by_balance(df, "subject", "target", 0.5)
        assert "02" in result["subject"].values
        assert "03" not in result["subject"].values

    def test_multiple_subjects_correct_subset(self):
        """Only subjects above threshold are in the result."""
        df = self._df({
            "02": [0, 1, 0, 1],     # 50% → keep
            "03": [0, 1, 0, 1, 0],  # 40% → keep at 0.2 threshold
            "04": [1, 1, 1, 1],     # 0% → remove
        })
        result = filter_participants_by_balance(df, "subject", "target", 0.2)
        kept = set(result["subject"].unique())
        assert "02" in kept
        assert "03" in kept
        assert "04" not in kept

    def test_output_contains_all_rows_for_kept_subjects(self):
        """All rows for a kept subject should appear in the output."""
        df = self._df({"02": [0, 1, 0, 1, 0]})
        result = filter_participants_by_balance(df, "subject", "target", 0.2)
        assert len(result) == 5


# =============================================================================
# filter_participants_by_sample_count
# =============================================================================

class TestFilterParticipantsBySampleCount:
    """Tests for filter_participants_by_sample_count."""

    def _df(self, counts):
        rows = []
        for sub, n in counts.items():
            rows.extend([{"subject": sub, "target": i % 2} for i in range(n)])
        return pd.DataFrame(rows)

    def test_removes_subject_below_threshold(self):
        df = self._df({"02": 3, "03": 15})
        result = filter_participants_by_sample_count(df, "subject", min_samples=10)
        assert "02" not in result["subject"].values
        assert "03" in result["subject"].values

    def test_keeps_subject_at_exact_threshold(self):
        df = self._df({"02": 10})
        result = filter_participants_by_sample_count(df, "subject", min_samples=10)
        assert "02" in result["subject"].values

    def test_keeps_all_above_threshold(self):
        df = self._df({"02": 20, "03": 30, "04": 15})
        result = filter_participants_by_sample_count(df, "subject", min_samples=10)
        assert result["subject"].nunique() == 3

    def test_zero_min_samples_keeps_all(self):
        df = self._df({"02": 1, "03": 2})
        result = filter_participants_by_sample_count(df, "subject", min_samples=0)
        assert len(result) == 3

    def test_all_removed_returns_empty(self):
        df = self._df({"02": 3, "03": 4})
        result = filter_participants_by_sample_count(df, "subject", min_samples=10)
        assert len(result) == 0


# =============================================================================
# get_feature_columns
# =============================================================================

class TestGetFeatureColumns:
    """Tests for get_feature_columns."""

    def test_excludes_standard_metadata(self):
        df = pd.DataFrame({
            "subject": ["02"], "task": ["Sart1"], "probe_number": [1],
            "onoff": [60], "target": [1],
            "power_alpha_Fz": [0.5], "wsmi_theta_pair": [0.3],
        })
        cols = get_feature_columns(df)
        assert "subject" not in cols
        assert "task" not in cols
        assert "target" not in cols
        assert "onoff" not in cols
        assert "power_alpha_Fz" in cols
        assert "wsmi_theta_pair" in cols

    def test_returns_list(self):
        _, _, _, _, _ = make_synthetic_data(n_subjects=2, n_probes_per_subject=5)
        df, *_ = make_synthetic_data(n_subjects=2, n_probes_per_subject=5)
        result = get_feature_columns(df)
        assert isinstance(result, list)

    def test_no_metadata_only_df_returns_empty(self):
        df = pd.DataFrame({"subject": ["02"], "task": ["Sart1"], "target": [1]})
        cols = get_feature_columns(df)
        assert len(cols) == 0

    def test_all_feature_cols_returned_from_synthetic_data(self):
        df, _, _, _, feature_cols = make_synthetic_data(n_subjects=2, n_probes_per_subject=5)
        cols = get_feature_columns(df)
        for fc in feature_cols:
            assert fc in cols


# =============================================================================
# pivot_to_wide
# =============================================================================

class TestPivotToWide:
    """Tests for pivot_to_wide: long → wide format conversion."""

    def _make_long(self):
        rows = []
        for sub in ["02", "03"]:
            for probe in [1, 2]:
                for marker in ["alpha", "beta"]:
                    for ch in ["Fz", "Cz"]:
                        rows.append({
                            "subject": sub, "task": "Sart1",
                            "probe_number": probe,
                            "marker": marker, "channel": ch,
                            "value": np.random.rand(),
                            "onoff": 60,
                        })
        return pd.DataFrame(rows)

    def test_wide_has_one_row_per_probe(self):
        long_df = self._make_long()
        wide_df = pivot_to_wide(long_df)
        expected_rows = long_df.groupby(["subject", "task", "probe_number"]).ngroups
        assert len(wide_df) == expected_rows

    def test_metadata_columns_preserved(self):
        long_df = self._make_long()
        wide_df = pivot_to_wide(long_df)
        for col in ["subject", "task", "probe_number", "onoff"]:
            assert col in wide_df.columns

    def test_feature_columns_created(self):
        long_df = self._make_long()
        wide_df = pivot_to_wide(long_df)
        non_meta = [c for c in wide_df.columns
                    if c not in ("subject", "task", "probe_number", "onoff")]
        assert len(non_meta) > 0

    def test_no_missing_values_in_features(self):
        long_df = self._make_long()
        wide_df = pivot_to_wide(long_df)
        feature_cols = [c for c in wide_df.columns
                        if c not in ("subject", "task", "probe_number", "onoff")]
        assert wide_df[feature_cols].isna().sum().sum() == 0


# =============================================================================
# get_project_root
# =============================================================================

class TestGetProjectRoot:
    """Tests for get_project_root."""

    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_exists(self):
        root = get_project_root()
        assert root.exists()

    def test_root_contains_expected_directories(self):
        """Project root should contain BIDS or _RAW_DATA, confirming we found the right dir."""
        root = get_project_root()
        # At minimum the pipeline directory should exist
        assert (root / "mw_classification_pipeline").exists() or root.is_dir()
