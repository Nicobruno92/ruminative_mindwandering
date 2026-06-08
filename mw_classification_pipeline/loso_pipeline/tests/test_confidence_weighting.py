"""
Tests for confidence-based sample weighting in the LOSO pipeline.

Covers:
- compute_within_subject_confidence_weights (within-subject min-max, floor,
  constant-confidence neutrality)
- Weight interpolation through within-subject SMOTE oversampling
- Pipeline plumbing of clf__sample_weight (enabled vs disabled equivalence)

All tests use synthetic data only — deterministic and fast.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.ml_utils import (
    compute_within_subject_confidence_weights,
    apply_within_subject_oversampling,
    run_model_pipeline_cv,
)
from conftest import make_synthetic_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_loso_classification import build_confidence_sample_weights


# =============================================================================
# compute_within_subject_confidence_weights
# =============================================================================

class TestComputeWithinSubjectConfidenceWeights:
    """Pure within-subject confidence-to-weight transform."""

    def test_within_subject_minmax_maps_to_floor_and_one(self):
        """Per subject: max confidence -> 1.0, min -> w_min, linear between."""
        # Subject A: confidences 0, 50, 100 -> with w_min=0.1 expect 0.1, 0.55, 1.0
        confidence = np.array([0.0, 50.0, 100.0])
        groups = np.array(["02", "02", "02"])

        weights = compute_within_subject_confidence_weights(
            confidence, groups, w_min=0.1, normalization="within_subject"
        )

        np.testing.assert_allclose(weights, [0.1, 0.55, 1.0])

    def test_constant_confidence_subject_is_neutral(self):
        """A subject with zero confidence range maps to all-ones (neutral)."""
        confidence = np.array([70.0, 70.0, 70.0])
        groups = np.array(["05", "05", "05"])

        weights = compute_within_subject_confidence_weights(
            confidence, groups, w_min=0.1, normalization="within_subject"
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0])

    def test_subjects_normalized_independently(self):
        """Each subject's range is normalized on its own scale, not globally."""
        # Subject 02 range 0-10, subject 03 range 90-100. Both span their own
        # min->max to w_min->1, despite very different absolute confidence.
        confidence = np.array([0.0, 10.0, 90.0, 100.0])
        groups = np.array(["02", "02", "03", "03"])

        weights = compute_within_subject_confidence_weights(
            confidence, groups, w_min=0.2, normalization="within_subject"
        )

        np.testing.assert_allclose(weights, [0.2, 1.0, 0.2, 1.0])

    def test_unknown_normalization_raises(self):
        """Unsupported normalization is rejected explicitly (no silent fallback)."""
        with pytest.raises(ValueError, match="normalization"):
            compute_within_subject_confidence_weights(
                np.array([1.0, 2.0]), np.array(["02", "02"]),
                normalization="zscore",
            )


# =============================================================================
# Weight interpolation through within-subject SMOTE
# =============================================================================

class TestWeightInterpolationThroughSmote:
    """apply_within_subject_oversampling carries and interpolates sample weights."""

    @pytest.fixture
    def imbalanced_single_subject(self):
        """One subject, 8 majority + 3 minority samples; distinct weights."""
        rng = np.random.default_rng(0)
        n_maj, n_min = 8, 3
        X = pd.DataFrame(rng.standard_normal((n_maj + n_min, 4)),
                         columns=[f"f{i}" for i in range(4)])
        y = np.array([0] * n_maj + [1] * n_min)
        groups = np.array(["07"] * (n_maj + n_min))
        # Weights in (0, 1], unique so we can detect leakage into features.
        weights = np.linspace(0.2, 1.0, n_maj + n_min)
        return X, y, groups, weights

    def test_returns_weights_aligned_to_expanded_set(self, imbalanced_single_subject):
        X, y, groups, weights = imbalanced_single_subject

        X_bal, y_bal, groups_bal, w_bal = apply_within_subject_oversampling(
            X, y, groups, method="SMOTE", k_neighbors=2, random_state=42,
            return_groups=True, weights=weights, return_weights=True,
        )

        assert len(w_bal) == len(y_bal)
        assert len(w_bal) == len(X_bal)
        # SMOTE balanced the single subject: 8 + 8 = 16 samples.
        assert len(y_bal) == 16

    def test_synthetic_weights_do_not_extrapolate(self, imbalanced_single_subject):
        """Interpolated weights stay within the original weight range."""
        X, y, groups, weights = imbalanced_single_subject

        _, _, _, w_bal = apply_within_subject_oversampling(
            X, y, groups, method="SMOTE", k_neighbors=2, random_state=42,
            return_groups=True, weights=weights, return_weights=True,
        )

        assert w_bal.min() >= weights.min() - 1e-9
        assert w_bal.max() <= weights.max() + 1e-9

    def test_weight_column_not_leaked_into_features(self, imbalanced_single_subject):
        """The carried weight must not appear as an extra feature column."""
        X, y, groups, weights = imbalanced_single_subject

        X_bal, _, _, _ = apply_within_subject_oversampling(
            X, y, groups, method="SMOTE", k_neighbors=2, random_state=42,
            return_groups=True, weights=weights, return_weights=True,
        )

        assert list(X_bal.columns) == list(X.columns)


# =============================================================================
# Pipeline plumbing of clf__sample_weight (LOSO)
# =============================================================================

class TestSampleWeightPlumbingLOSO:
    """run_model_pipeline_cv threads confidence weights to clf.fit (LOSO)."""

    @pytest.fixture
    def data(self):
        df, X, y, groups, _ = make_synthetic_data(
            n_subjects=5, n_probes_per_subject=24, n_features=10
        )
        rng = np.random.default_rng(1)
        weights = compute_within_subject_confidence_weights(
            rng.integers(0, 101, size=len(y)).astype(float), groups.values,
            w_min=0.1, normalization="within_subject",
        )
        return X, y, groups, weights

    def test_runs_with_sample_weights_within_scope(self, data):
        """rf + within-subject SMOTE + sample_weights produces valid metrics."""
        X, y, groups, weights = data

        result = run_model_pipeline_cv(
            X, y, groups, model_type="rf",
            use_smote=True, oversampling_scope="within",
            feature_selection_method="none", scale_by_participant="none",
            cv_n_jobs=1, fixed_random_state=42,
            sample_weights=weights,
        )

        assert 0.0 <= result["mean_auc"].iloc[0] <= 1.0

    def test_sample_weights_with_global_scope_raises(self, data):
        """Global SMOTE lives inside the pipeline and cannot carry weights."""
        X, y, groups, weights = data

        with pytest.raises(ValueError, match="oversampling_scope"):
            run_model_pipeline_cv(
                X, y, groups, model_type="rf",
                use_smote=True, oversampling_scope="global",
                feature_selection_method="none", scale_by_participant="none",
                cv_n_jobs=1, fixed_random_state=42,
                sample_weights=weights,
            )

    def test_sample_weights_with_oneclass_raises(self, data):
        """One-class models do not use labels, so weighting is undefined."""
        X, y, groups, weights = data

        with pytest.raises(ValueError, match="one-class|sample_weight"):
            run_model_pipeline_cv(
                X, y, groups, model_type="iforest",
                use_smote=False, oversampling_scope="within",
                feature_selection_method="none", scale_by_participant="none",
                cv_n_jobs=1, fixed_random_state=42,
                sample_weights=weights,
            )


# =============================================================================
# Config wrapper: build_confidence_sample_weights
# =============================================================================

class TestBuildConfidenceSampleWeights:
    """Reads the contrast's confidence_weight block and builds weights."""

    @pytest.fixture
    def df_groups(self):
        df, _, _, groups, _ = make_synthetic_data(n_subjects=4, n_probes_per_subject=10)
        rng = np.random.default_rng(2)
        df = df.copy()
        df["confidence"] = rng.integers(0, 101, size=len(df)).astype(float)
        return df, groups

    def test_absent_block_returns_none(self, df_groups):
        df, groups = df_groups
        config = {"label_contrasts": {"ON_vs_OFF": {}}}
        assert build_confidence_sample_weights(config, "ON_vs_OFF", df, groups) is None

    def test_disabled_block_returns_none(self, df_groups):
        df, groups = df_groups
        config = {"label_contrasts": {"ON_vs_OFF": {
            "confidence_weight": {"enabled": False}}}}
        assert build_confidence_sample_weights(config, "ON_vs_OFF", df, groups) is None

    def test_enabled_returns_weights_in_range(self, df_groups):
        df, groups = df_groups
        config = {"label_contrasts": {"ON_vs_OFF": {
            "confidence_weight": {"enabled": True, "w_min": 0.3}}}}
        w = build_confidence_sample_weights(config, "ON_vs_OFF", df, groups)
        assert w is not None
        assert len(w) == len(df)
        assert w.min() >= 0.3 - 1e-9
        assert w.max() <= 1.0 + 1e-9

    def test_enabled_without_confidence_column_raises(self, df_groups):
        df, groups = df_groups
        df = df.drop(columns=["confidence"])
        config = {"label_contrasts": {"ON_vs_OFF": {
            "confidence_weight": {"enabled": True}}}}
        with pytest.raises(ValueError, match="confidence"):
            build_confidence_sample_weights(config, "ON_vs_OFF", df, groups)

    def test_enabled_with_nan_confidence_raises(self, df_groups):
        df, groups = df_groups
        df = df.copy()
        df.loc[df.index[0], "confidence"] = np.nan
        config = {"label_contrasts": {"ON_vs_OFF": {
            "confidence_weight": {"enabled": True}}}}
        with pytest.raises(ValueError, match="NaN"):
            build_confidence_sample_weights(config, "ON_vs_OFF", df, groups)
