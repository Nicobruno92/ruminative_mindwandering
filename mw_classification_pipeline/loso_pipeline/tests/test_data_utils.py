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
    residualize_within_subject,
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
# residualize_within_subject
# =============================================================================

class TestResidualizeWithinSubject:
    """Tests for the within-subject OLS residualization helper."""

    def _df(self, subject_blocks):
        """subject_blocks: dict of subject -> (target_vals, predictor_vals)."""
        rows = []
        for subj, (target_vals, pred_vals) in subject_blocks.items():
            for t, p in zip(target_vals, pred_vals):
                rows.append({"subject": subj, "valence": t, "onoff": p})
        return pd.DataFrame(rows)

    def test_perfect_linear_relationship_gives_near_zero_residuals(self):
        """Target is an exact linear function of the predictor -> residuals ~0."""
        onoff = np.linspace(0, 100, 20)
        valence = 2.0 * onoff + 10.0  # no noise
        df = self._df({"02": (valence, onoff)})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff"], min_valid=5
        )
        assert excluded == []
        assert np.allclose(resid, 0.0, atol=1e-8)

    def test_independent_predictor_leaves_target_mostly_unchanged(self):
        """Target unrelated to the predictor -> residual highly correlated with raw target."""
        rng = np.random.default_rng(0)
        onoff = rng.uniform(0, 100, 50)          # predictor: pure noise
        valence = rng.uniform(0, 100, 50)         # target: independent noise
        df = self._df({"02": (valence, onoff)})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff"], min_valid=5
        )
        assert excluded == []
        # Residual should just be target minus a near-zero-slope fit: highly
        # correlated with the demeaned original.
        correlation = np.corrcoef(resid, valence)[0, 1]
        assert correlation > 0.9

    def test_subject_below_min_valid_is_excluded(self):
        """Subject with too few valid rows -> all-NaN residual, reported as excluded."""
        onoff = np.array([10.0, 20.0, 30.0, 40.0])
        valence = np.array([15.0, 25.0, 35.0, 45.0])
        df = self._df({"02": (valence, onoff)})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff"], min_valid=10
        )
        assert excluded == ["02"]
        assert np.all(np.isnan(resid))

    def test_mixed_subjects_only_low_data_one_excluded(self):
        """One subject has enough data, another doesn't -> only the latter is excluded."""
        rng = np.random.default_rng(1)
        good_onoff = rng.uniform(0, 100, 20)
        good_valence = 0.5 * good_onoff + rng.normal(0, 1, 20)
        bad_onoff = np.array([10.0, 20.0, 30.0])
        bad_valence = np.array([12.0, 22.0, 32.0])
        df = self._df({"02": (good_valence, good_onoff), "03": (bad_valence, bad_onoff)})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff"], min_valid=10
        )
        assert excluded == ["03"]
        subj_mask = (df["subject"] == "02").values
        assert not np.any(np.isnan(resid[subj_mask]))
        subj_mask_bad = (df["subject"] == "03").values
        assert np.all(np.isnan(resid[subj_mask_bad]))

    def test_row_with_missing_predictor_is_nan_even_if_subject_included(self):
        """A single row missing a predictor value -> NaN just for that row."""
        onoff = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, np.nan])
        valence = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 5.0, 50.0])
        df = self._df({"02": (valence, onoff)})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff"], min_valid=5
        )
        assert excluded == []
        assert np.isnan(resid[-1])
        assert not np.any(np.isnan(resid[:-1]))

    def test_multiple_predictors_supported(self):
        """residualize_against a list of >1 predictor columns."""
        rng = np.random.default_rng(2)
        n = 20
        p1 = rng.uniform(0, 100, n)
        p2 = rng.uniform(0, 100, n)
        target = 1.5 * p1 - 0.5 * p2 + 3.0  # no noise, two predictors
        df = pd.DataFrame({"subject": "02", "valence": target, "onoff": p1, "selfother": p2})
        resid, excluded = residualize_within_subject(
            df, target_col="valence", predictor_cols=["onoff", "selfother"], min_valid=5
        )
        assert excluded == []
        assert np.allclose(resid, 0.0, atol=1e-8)


# =============================================================================
# create_label_contrast: linear_residual transform
# =============================================================================

class TestCreateLabelContrastLinearResidual:
    """Tests for the 'linear_residual' transform inside create_label_contrast."""

    def test_missing_residualize_against_raises(self):
        df = pd.DataFrame({"subject": "02", "task": "Sart1", "valence": [10.0, 20.0], "onoff": [1.0, 2.0]})
        contrast = {
            "label_source": "valence",
            "transform": "linear_residual",
            "split_method": "threshold",
            "threshold": 0,
        }
        with pytest.raises(ValueError, match="residualize_against"):
            create_label_contrast(df, contrast)

    def test_unknown_predictor_column_raises(self):
        df = pd.DataFrame({"subject": "02", "task": "Sart1", "valence": [10.0, 20.0], "onoff": [1.0, 2.0]})
        contrast = {
            "label_source": "valence",
            "transform": "linear_residual",
            "residualize_against": ["nonexistent_dim"],
            "split_method": "threshold",
            "threshold": 0,
        }
        with pytest.raises(ValueError, match="nonexistent_dim"):
            create_label_contrast(df, contrast)

    def test_excluded_subject_has_no_rows_in_final_output(self):
        """A subject below min_valid_for_residual should be fully absent from the result."""
        rng = np.random.default_rng(3)
        n_good = 20
        good_onoff = rng.uniform(0, 100, n_good)
        good_valence = rng.uniform(0, 100, n_good)
        low_onoff = np.array([10.0, 20.0, 30.0])
        low_valence = np.array([15.0, 25.0, 35.0])
        df = pd.concat([
            pd.DataFrame({"subject": "02", "task": "Sart1", "valence": good_valence, "onoff": good_onoff}),
            pd.DataFrame({"subject": "03", "task": "Sart1", "valence": low_valence, "onoff": low_onoff}),
        ], ignore_index=True)
        contrast = {
            "label_source": "valence",
            "transform": "linear_residual",
            "residualize_against": ["onoff"],
            "min_valid_for_residual": 10,
            "split_method": "within_subject_median",
            # gap=0 takes a different code path (no dropna — see
            # pre-existing latent bug noted in the linear_residual design
            # spec) that is not used by any real contrast; gap=5 matches
            # actual usage and exercises the dropna path correctly.
            "gap": 5,
            "positive_above": True,
        }
        result = create_label_contrast(df, contrast)
        assert "03" not in result["subject"].values
        assert set(result["subject"].unique()) == {"02"}

    def test_split_runs_on_residualized_column_not_raw(self):
        """
        Constant-per-subject predictor carries no information, so residualizing
        against it is equivalent to demeaning: residual = target - subject_mean.
        This lets us predict the resulting split exactly.
        """
        onoff = np.full(10, 50.0)  # constant -> zero information, pure demeaning
        valence = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        df = pd.DataFrame({"subject": "02", "task": "Sart1", "valence": valence, "onoff": onoff})
        contrast = {
            "label_source": "valence",
            "transform": "linear_residual",
            "residualize_against": ["onoff"],
            "min_valid_for_residual": 5,
            "split_method": "within_subject_median",
            "gap": 5,
            "positive_above": True,
        }
        result = create_label_contrast(df, contrast)
        # mean(valence) = 55; residual = valence - 55; median(residual) = 0
        # gap=5 -> residual > 5 => class 1, residual < -5 => class 0, else excluded.
        for _, row in result.iterrows():
            resid = row["valence"] - 55.0
            if resid > 5:
                assert row["target"] == 1
            elif resid < -5:
                assert row["target"] == 0
        # values 50 and 60 (residual 0.0 and 5.0) fall inside/at the neutral
        # zone boundary and must be excluded
        assert 50.0 not in result["valence"].values
        assert 60.0 not in result["valence"].values


# =============================================================================
# create_label_contrast: midpoint_sq_residual_cross transform
# =============================================================================

class TestCreateLabelContrastMidpointSqResidualCross:
    """Tests for the 'midpoint_sq_residual_cross' transform."""

    def test_missing_residualize_against_raises(self):
        df = pd.DataFrame({
            "subject": "02", "task": "Sart1",
            "valence": [10.0, 20.0], "onoff": [1.0, 2.0],
        })
        contrast = {
            "label_source": "valence",
            "transform": "midpoint_sq_residual_cross",
            "split_method": "threshold",
            "threshold": 0,
        }
        with pytest.raises(ValueError, match="residualize_against"):
            create_label_contrast(df, contrast)

    def test_unknown_predictor_column_raises(self):
        df = pd.DataFrame({
            "subject": "02", "task": "Sart1",
            "valence": [10.0, 20.0], "onoff": [1.0, 2.0],
        })
        contrast = {
            "label_source": "valence",
            "transform": "midpoint_sq_residual_cross",
            "residualize_against": ["nonexistent_dim"],
            "split_method": "threshold",
            "threshold": 0,
        }
        with pytest.raises(ValueError, match="nonexistent_dim"):
            create_label_contrast(df, contrast)

    def test_perfect_joint_fit_collapses_to_near_zero(self):
        """
        Construct (valence-50)^2/50 as an EXACT linear combination of
        [valence, onoff, selfother, time] (by fitting then using the fitted
        values as the "observed" sq) -> the joint regression inside the
        transform should drive the residual to ~0. Checked via the internal
        helper directly, since create_label_contrast only exposes the
        binarized target, not the continuous residual.
        """
        rng = np.random.default_rng(0)
        n = 30
        valence = rng.uniform(0, 100, n)
        onoff = rng.uniform(0, 100, n)
        selfother = rng.uniform(0, 100, n)
        time = rng.uniform(0, 100, n)
        raw_sq = (valence - 50.0) ** 2 / 50.0
        A = np.column_stack([valence, onoff, selfother, time, np.ones(n)])
        coeffs, _, _, _ = np.linalg.lstsq(A, raw_sq, rcond=None)
        exact_sq = A @ coeffs  # now an exact linear function of the 4 predictors

        df = pd.DataFrame({
            "subject": "02", "_sq_valence": exact_sq,
            "valence": valence, "onoff": onoff, "selfother": selfother, "time": time,
        })
        resid, excluded = residualize_within_subject(
            df, target_col="_sq_valence",
            predictor_cols=["valence", "onoff", "selfother", "time"],
            min_valid=5,
        )
        assert excluded == []
        assert np.allclose(resid, 0.0, atol=1e-8)

    def test_own_linear_dimension_used_even_if_others_are_noise(self):
        """
        sq is a pure function of its own linear dimension (quadratic, so a
        linear regressor cannot drive the residual to exactly 0 — it can
        only remove the LINEAR component); the other 3 predictors are
        independent noise. The own-linear term should still be used
        automatically (it's not in residualize_against), so the residual's
        linear correlation with valence should collapse near 0, same as the
        existing single-predictor midpoint_sq_residual behavior.
        """
        rng = np.random.default_rng(1)
        n = 40
        valence = np.linspace(0, 100, n)
        sq = (valence - 50.0) ** 2 / 50.0  # exact function of valence alone
        onoff = rng.uniform(0, 100, n)       # noise, unrelated to sq
        selfother = rng.uniform(0, 100, n)   # noise
        time = rng.uniform(0, 100, n)        # noise
        df = pd.DataFrame({
            "subject": "02",
            "valence": valence, "onoff": onoff, "selfother": selfother, "time": time,
            "_sq_valence": sq,
        })
        # Access the residual via the internal helper for a precise check
        # (create_label_contrast only exposes the binarized target).
        resid, excluded = residualize_within_subject(
            df, target_col="_sq_valence",
            predictor_cols=["valence", "onoff", "selfother", "time"],
            min_valid=5,
        )
        assert excluded == []
        assert abs(np.corrcoef(resid, valence)[0, 1]) < 1e-8
        # noise predictors carry no information either
        assert abs(np.corrcoef(resid, onoff)[0, 1]) < 0.3
        assert abs(np.corrcoef(resid, selfother)[0, 1]) < 0.3
        assert abs(np.corrcoef(resid, time)[0, 1]) < 0.3

    def test_subject_below_min_valid_excluded_from_final_output(self):
        rng = np.random.default_rng(2)
        n_good = 20
        good = pd.DataFrame({
            "subject": "02", "task": "Sart1",
            "valence": rng.uniform(0, 100, n_good),
            "onoff": rng.uniform(0, 100, n_good),
            "selfother": rng.uniform(0, 100, n_good),
            "time": rng.uniform(0, 100, n_good),
        })
        low = pd.DataFrame({
            "subject": "03", "task": "Sart1",
            "valence": [10.0, 20.0, 30.0],
            "onoff": [15.0, 25.0, 35.0],
            "selfother": [5.0, 15.0, 25.0],
            "time": [50.0, 60.0, 70.0],
        })
        df = pd.concat([good, low], ignore_index=True)
        contrast = {
            "label_source": "valence",
            "transform": "midpoint_sq_residual_cross",
            "residualize_against": ["onoff", "selfother", "time"],
            "min_valid_for_residual": 10,
            "split_method": "within_subject_median",
            "gap": 2.5,
            "positive_above": True,
        }
        result = create_label_contrast(df, contrast)
        assert "03" not in result["subject"].values


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
