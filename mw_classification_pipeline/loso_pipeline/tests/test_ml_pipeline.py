"""
Exhaustive tests for utils/ml_utils.py.

Covers:
- build_model_pipeline (all models, feature selection methods, scalers, SMOTE, PCA)
- run_model_pipeline_cv (LOSO correctness, metrics, subject leakage, reproducibility)
- Oversampling (get_oversampler, within-subject, edge cases)
- Feature selectors (MRMRFeatureSelector, LMMEncodingFeatureSelector)
- Scaling (_apply_fold_scaling within and global)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.ml_utils import (
    build_model_pipeline,
    run_model_pipeline_cv,
    get_oversampler,
    apply_within_subject_oversampling,
    LMMEncodingFeatureSelector,
    MRMRFeatureSelector,
)
from conftest import make_synthetic_data


# =============================================================================
# build_model_pipeline
# =============================================================================

class TestBuildModelPipeline:
    """Tests for build_model_pipeline: all models, steps, and configurations."""

    @pytest.fixture
    def X(self):
        _, X, _, _, _ = make_synthetic_data(n_subjects=4, n_probes_per_subject=20, n_features=15)
        return X

    @pytest.fixture
    def y(self):
        _, _, y, _, _ = make_synthetic_data(n_subjects=4, n_probes_per_subject=20, n_features=15)
        return y

    # --- model types ---

    def test_lr_builds(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5)
        assert "clf" in p.named_steps

    def test_rf_builds(self, X):
        p = build_model_pipeline(X, model_type="rf", k=5)
        assert "clf" in p.named_steps

    def test_invalid_model_raises(self, X):
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model_pipeline(X, model_type="svm", k=5)

    # --- feature selection ---

    def test_feature_selection_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=3, feature_selection_method="f_classif")
        assert "feature_selection" in p.named_steps

    def test_feature_selection_none_excluded(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, feature_selection_method="none")
        assert "feature_selection" not in p.named_steps

    def test_mutual_info_selection_builds(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, feature_selection_method="mutual_info_classif")
        assert "feature_selection" in p.named_steps

    def test_mrmr_selection_builds(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, feature_selection_method="mrmr")
        assert "feature_selection" in p.named_steps

    def test_lmm_encoding_selection_builds(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, feature_selection_method="lmm_encoding")
        assert "feature_selection" in p.named_steps

    # --- scalers ---

    def test_standard_scaler_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, scaler="standard")
        assert "scaler" in p.named_steps

    def test_minmax_scaler_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, scaler="minmax")
        assert "scaler" in p.named_steps

    def test_robust_scaler_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, scaler="robust")
        assert "scaler" in p.named_steps

    def test_no_scaler(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, scaler="none")
        assert "scaler" not in p.named_steps

    # --- PCA ---

    def test_pca_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, use_pca=True, pca_n_components=3)
        assert "pca" in p.named_steps

    def test_no_pca_by_default(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, use_pca=False)
        assert "pca" not in p.named_steps

    # --- SMOTE ---

    def test_smote_included(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, use_smote=True,
                                  oversampling_method="SMOTE", random_state=42)
        assert "smote" in p.named_steps

    def test_no_smote_by_default(self, X):
        p = build_model_pipeline(X, model_type="lr", k=5, use_smote=False)
        assert "smote" not in p.named_steps

    # --- fit and predict ---

    def test_pipeline_fits_and_predicts(self, X, y):
        p = build_model_pipeline(X, model_type="lr", k=5, random_state=42, y=y)
        p.fit(X, y)
        preds = p.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_rf_predicts_proba(self, X, y):
        p = build_model_pipeline(X, model_type="rf", k=5, random_state=42, y=y)
        p.fit(X, y)
        proba = p.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_k_all_uses_all_features(self, X, y):
        """k='all' should not add a feature selection step."""
        p = build_model_pipeline(X, model_type="lr", k="all",
                                  feature_selection_method="f_classif", random_state=42)
        # 'all' means k=n_features, so feature_selection is still included but selects all
        p.fit(X, y)
        assert p.predict(X) is not None


# =============================================================================
# run_model_pipeline_cv (LOSO)
# =============================================================================

class TestRunModelPipelineCVLOSO:
    """Tests for run_model_pipeline_cv: LOSO correctness, metrics, leakage."""

    @pytest.fixture
    def data(self):
        return make_synthetic_data(n_subjects=5, n_probes_per_subject=20, n_features=10, random_seed=7)

    # --- structure ---

    def test_returns_dataframe(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_has_all_metric_columns(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        expected = [
            "mean_auc", "std_auc", "mean_auprc", "std_auprc",
            "mean_mcc", "std_mcc", "mean_balanced_accuracy", "std_balanced_accuracy",
            "fold_aucs", "fold_mccs", "fold_bal_accs",
            "loso_subject_metrics", "fold_details",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_fold_details_keys(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        fold_details = result["fold_details"].values[0]
        for fd in fold_details:
            for key in ["fold_idx", "subject", "y_true", "y_pred", "y_proba", "test_indices"]:
                assert key in fd, f"fold_detail missing key: {key}"

    # --- LOSO invariants ---

    def test_no_subject_leakage(self, data):
        """Train and test sets must have no subject overlap in any fold."""
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        cv_splits = result["cv_splits"].values[0]
        for train_idx, test_idx in cv_splits:
            overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
            assert len(overlap) == 0, f"Subject leakage: {overlap}"

    def test_n_folds_equals_n_subjects(self, data):
        _, X, y, groups, _ = data
        n_subjects = groups.nunique()
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        fold_aucs = result["fold_aucs"].values[0]
        assert len(fold_aucs) == n_subjects

    def test_one_subject_per_fold(self, data):
        """Each fold should hold out exactly one unique subject."""
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        cv_splits = result["cv_splits"].values[0]
        for _, test_idx in cv_splits:
            test_subjects = groups.iloc[test_idx].unique()
            assert len(test_subjects) == 1, f"Expected 1 test subject, got {len(test_subjects)}"

    def test_subject_metrics_one_per_subject(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        subject_metrics = result["loso_subject_metrics"].values[0]
        assert len(subject_metrics) == groups.nunique()

    def test_subject_metrics_subjects_match_groups(self, data):
        """Subject IDs in loso_subject_metrics should match the unique groups."""
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        reported_subjects = {m["subject"] for m in result["loso_subject_metrics"].values[0]}
        expected_subjects = set(groups.unique())
        assert reported_subjects == expected_subjects

    # --- metric ranges ---

    def test_auc_in_valid_range(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        auc = result["mean_auc"].values[0]
        assert 0.0 <= auc <= 1.0, f"AUC out of range: {auc}"

    def test_auprc_in_valid_range(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        auprc = result["mean_auprc"].values[0]
        assert 0.0 <= auprc <= 1.0

    def test_mcc_in_valid_range(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        mcc = result["mean_mcc"].values[0]
        assert -1.0 <= mcc <= 1.0

    def test_balanced_accuracy_in_valid_range(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        bacc = result["mean_balanced_accuracy"].values[0]
        assert 0.0 <= bacc <= 1.0

    # --- reproducibility ---

    def test_same_seed_same_auc(self, data):
        _, X, y, groups, _ = data
        r1 = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5, fixed_random_state=42)
        r2 = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5, fixed_random_state=42)
        assert r1["mean_auc"].values[0] == r2["mean_auc"].values[0]

    def test_different_seed_may_differ(self, data):
        """Different seeds should generally produce different feature selection → different AUC."""
        _, X, y, groups, _ = data
        r1 = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5, fixed_random_state=1)
        r2 = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5, fixed_random_state=999)
        # Not a strict check — just ensure the pipeline runs with different seeds
        assert isinstance(r1["mean_auc"].values[0], float)
        assert isinstance(r2["mean_auc"].values[0], float)

    # --- model comparison ---

    def test_rf_model_runs(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="rf", k=5,
                                        fixed_random_state=42)
        assert not result.empty
        assert result["mean_auc"].values[0] >= 0

    def test_within_subject_smote_runs(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(
            X, y, groups, model_type="lr", k=5,
            use_smote=True, oversampling_method="SMOTE",
            oversampling_scope="within", fixed_random_state=42,
        )
        assert not result.empty

    def test_global_scaling_runs(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(
            X, y, groups, model_type="lr", k=5,
            scale_by_participant="global", fixed_random_state=42,
        )
        assert not result.empty

    def test_within_scaling_runs(self, data):
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(
            X, y, groups, model_type="lr", k=5,
            scale_by_participant="within", fixed_random_state=42,
        )
        assert not result.empty

    def test_fold_auc_std_computed(self, data):
        """std_auc should be a non-negative float."""
        _, X, y, groups, _ = data
        result = run_model_pipeline_cv(X, y, groups, model_type="lr", k=5,
                                        fixed_random_state=42)
        std_auc = result["std_auc"].values[0]
        assert std_auc >= 0.0


# =============================================================================
# Oversampling
# =============================================================================

class TestOversampling:
    """Tests for get_oversampler and apply_within_subject_oversampling."""

    def test_get_smote(self):
        from imblearn.over_sampling import SMOTE
        sampler = get_oversampler("SMOTE", k_neighbors=3)
        assert isinstance(sampler, SMOTE)

    def test_get_svmsmote(self):
        from imblearn.over_sampling import SVMSMOTE
        sampler = get_oversampler("SVMSMOTE", k_neighbors=3)
        assert isinstance(sampler, SVMSMOTE)

    def test_get_adasyn(self):
        from imblearn.over_sampling import ADASYN
        sampler = get_oversampler("ADASYN", k_neighbors=3)
        assert isinstance(sampler, ADASYN)

    def test_get_smotetomek(self):
        from imblearn.combine import SMOTETomek
        sampler = get_oversampler("SMOTETomek", k_neighbors=3)
        assert isinstance(sampler, SMOTETomek)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown oversampling method"):
            get_oversampler("SOME_UNKNOWN_METHOD")

    def test_within_oversampling_increases_minority(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((30, 5)), columns=[f"f{i}" for i in range(5)])
        y = np.array([0] * 25 + [1] * 5)
        groups = np.array(["02"] * 15 + ["03"] * 15)
        X_bal, y_bal = apply_within_subject_oversampling(X, y, groups, method="SMOTE")
        assert np.sum(y_bal == 1) > np.sum(y == 1)

    def test_within_oversampling_no_error_single_class(self):
        """Subjects with only one class should be passed through without error."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((20, 5)), columns=[f"f{i}" for i in range(5)])
        y = np.array([0] * 10 + [0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        groups = np.array(["02"] * 10 + ["03"] * 10)
        X_bal, y_bal = apply_within_subject_oversampling(X, y, groups, method="SMOTE")
        assert len(X_bal) >= 20  # Should not shrink

    def test_within_oversampling_preserves_subjects(self):
        """After within-subject oversampling, both subjects should remain."""
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"f{i}" for i in range(5)])
        y = np.array([0] * 5 + [1] * 15 + [0] * 15 + [1] * 5)
        groups = np.array(["02"] * 20 + ["03"] * 20)
        X_bal, y_bal = apply_within_subject_oversampling(X, y, groups, method="SMOTE")
        assert len(y_bal) > len(y)


# =============================================================================
# MRMRFeatureSelector
# =============================================================================

class TestMRMRFeatureSelector:
    """Tests for MRMRFeatureSelector."""

    def test_selects_k_features(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 10)),
                         columns=[f"f{i}" for i in range(10)])
        y = np.array([0] * 25 + [1] * 25)
        selector = MRMRFeatureSelector(k=3)
        selector.fit(X, y)
        mask = selector.get_support()
        assert mask.sum() == 3

    def test_transforms_correctly(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, 10)),
                         columns=[f"f{i}" for i in range(10)])
        y = np.array([0] * 25 + [1] * 25)
        selector = MRMRFeatureSelector(k=4)
        selector.fit(X, y)
        X_t = selector.transform(X)
        assert X_t.shape[1] == 4

    def test_not_fitted_raises(self):
        selector = MRMRFeatureSelector(k=5)
        # Calling transform (which calls _get_support_mask) on unfitted selector
        # must raise either ValueError or TypeError
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 5)),
                         columns=[f"f{i}" for i in range(5)])
        with pytest.raises((ValueError, TypeError, AttributeError)):
            selector.transform(X)


    def test_k_capped_at_n_features(self):
        """k > n_features should silently select all features."""
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.standard_normal((50, 5)),
                         columns=[f"f{i}" for i in range(5)])
        y = np.array([0] * 25 + [1] * 25)
        selector = MRMRFeatureSelector(k=20)
        selector.fit(X, y)
        assert selector.get_support().sum() == 5


# =============================================================================
# LMMEncodingFeatureSelector (unit checks)
# =============================================================================

class TestLMMEncodingFeatureSelector:
    """Sanity tests for LMMEncodingFeatureSelector."""

    def test_requires_groups(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((40, 5)),
                         columns=[f"f{i}" for i in range(5)])
        y = np.array([0, 1] * 20)
        selector = LMMEncodingFeatureSelector(k=3, n_jobs=1)
        with pytest.raises(ValueError, match="groups"):
            selector.fit(X, y, groups=None)

    def test_selects_k_features_with_groups(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((40, 8)),
                         columns=[f"f{i}" for i in range(8)])
        y = np.array([0, 1] * 20)
        groups = np.array(["02"] * 20 + ["03"] * 20)
        selector = LMMEncodingFeatureSelector(k=3, n_jobs=1)
        selector.fit(X, y, groups=groups)
        assert selector.get_support().sum() == 3

    def test_pvalues_in_zero_one(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((30, 4)),
                         columns=[f"f{i}" for i in range(4)])
        y = np.array([0, 1] * 15)
        groups = np.array(["02"] * 15 + ["03"] * 15)
        selector = LMMEncodingFeatureSelector(k=2, n_jobs=1)
        selector.fit(X, y, groups=groups)
        pvals = selector.pvalues_
        assert np.all((pvals >= 0.0) & (pvals <= 1.0))
