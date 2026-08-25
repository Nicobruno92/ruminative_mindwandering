"""
Tests for `_extract_fold_importances` — recovering Gini importances from a
fitted fold pipeline.

The regression these pin down: the original code only populated the importance
vector when the pipeline had a `feature_selection` step. Both configs now set
`feature_selection.method: "none"`, so that branch stopped firing and every
`*_feature_importances.csv` on disk was written as all zeros — silently, because
an all-zero vector is a valid array and every downstream consumer accepted it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.ml_utils import _extract_fold_importances

N_FEATURES = 12
RANDOM_STATE = 0


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    """A small separable binary problem with informative leading columns."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(80, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _fit(steps: list, data: tuple[np.ndarray, np.ndarray]) -> Pipeline:
    X, y = data
    pipeline = Pipeline(steps)
    pipeline.fit(X, y)
    return pipeline


class TestWithoutFeatureSelection:
    """`feature_selection.method: "none"` — the current production shape."""

    def test_importances_are_not_all_zero(self, data):
        pipeline = _fit(
            [("scaler", StandardScaler()),
             ("clf", RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        assert result.shape == (N_FEATURES,)
        assert result.sum() > 0, "the exact regression: an all-zero vector on every run"

    def test_matches_classifier_vector_exactly(self, data):
        pipeline = _fit(
            [("clf", RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        np.testing.assert_array_equal(result, pipeline.named_steps["clf"].feature_importances_)

    def test_informative_features_rank_highest(self, data):
        """Sanity check that the vector is positioned, not merely non-zero."""
        pipeline = _fit(
            [("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        assert set(np.argsort(result)[-2:]) == {0, 1}


class TestWithFeatureSelection:
    """Selection active — importances must be scattered back to original columns."""

    def test_scattered_to_selected_positions(self, data):
        k = 4
        pipeline = _fit(
            [("feature_selection", SelectKBest(f_classif, k=k)),
             ("clf", RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        selected = pipeline.named_steps["feature_selection"].get_support(indices=True)

        assert result.shape == (N_FEATURES,)
        assert np.count_nonzero(result) <= k
        unselected = np.setdiff1d(np.arange(N_FEATURES), selected)
        np.testing.assert_array_equal(result[unselected], np.zeros(len(unselected)))
        np.testing.assert_allclose(
            result[selected], pipeline.named_steps["clf"].feature_importances_
        )

    def test_total_importance_preserved(self, data):
        pipeline = _fit(
            [("feature_selection", SelectKBest(f_classif, k=5)),
             ("clf", RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        assert result.sum() == pytest.approx(1.0)


class TestUninterpretableShapes:
    """Cases where no honest feature-space mapping exists."""

    def test_pca_returns_zeros(self, data):
        """After PCA the importances index components, not features."""
        pipeline = _fit(
            [("pca", PCA(n_components=3, random_state=RANDOM_STATE)),
             ("clf", RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE))],
            data,
        )
        result = _extract_fold_importances(pipeline, N_FEATURES)
        np.testing.assert_array_equal(result, np.zeros(N_FEATURES))

    def test_classifier_without_importances_returns_zeros(self, data):
        from sklearn.linear_model import LogisticRegression

        pipeline = _fit([("clf", LogisticRegression(max_iter=500))], data)
        result = _extract_fold_importances(pipeline, N_FEATURES)
        np.testing.assert_array_equal(result, np.zeros(N_FEATURES))

    def test_length_mismatch_raises(self, data):
        """
        No selection step, yet the widths disagree — that is a broken pipeline,
        so it must surface rather than be padded into a plausible-looking vector.
        """
        pipeline = _fit(
            [("clf", RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE))],
            data,
        )
        with pytest.raises(ValueError, match="no feature_selection step"):
            _extract_fold_importances(pipeline, N_FEATURES + 5)
