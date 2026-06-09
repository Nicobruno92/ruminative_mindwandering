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


from utils import build_subject_level_features, build_probe_level_features, build_block_level_features


# --- Synthetic helpers ---

def _make_probe_df(n_subjects: int = 6, probes_per_subject: int = 4) -> pd.DataFrame:
    """Minimal synthetic DataFrame mimicking probe_level_aggregated_data."""
    rng = np.random.RandomState(0)
    rows = []
    for s_idx in range(n_subjects):
        subj = str(s_idx + 2).zfill(2)
        group = "Controls" if s_idx < 4 else "Risk of Depression"
        for t_idx, task in enumerate(["Sart1", "Sart2", "Sart3", "Sart4"]):
            ie = "baseline" if task in ["Sart1", "Sart3"] else ("inclusion" if s_idx % 2 == 0 else "exclusion")
            for p in range(probes_per_subject):
                rows.append({
                    "subject": int(subj),
                    "task": task,
                    "probe_number": float(p + 1),
                    "onoff": rng.uniform(0, 100),
                    "valence": rng.uniform(0, 100),
                    "selfother": rng.uniform(0, 100),
                    "time": rng.uniform(0, 100),
                    "confidence": rng.uniform(0, 100),
                    "omission_rate": rng.uniform(0, 0.5),
                    "commission_rate": rng.uniform(0, 0.5),
                    "rt_mean": rng.uniform(200, 600),
                    "rtcv": rng.uniform(0, 0.5),
                    "PC1": rng.randn() if rng.rand() > 0.5 else np.nan,
                    "PC2": rng.randn() if rng.rand() > 0.5 else np.nan,
                    "PC3": rng.randn() if rng.rand() > 0.5 else np.nan,
                    "group": group,
                    "inclusion_exclusion": ie,
                })
    return pd.DataFrame(rows)


FEATURE_COLS = ["onoff", "valence", "selfother", "time", "confidence",
                "omission_rate", "commission_rate", "rt_mean", "rtcv",
                "PC1", "PC2", "PC3"]


def test_build_subject_level_shapes():
    df = _make_probe_df(n_subjects=6, probes_per_subject=4)
    X, y, groups = build_subject_level_features(df, FEATURE_COLS, "group", "Risk of Depression")
    assert X.shape == (6, len(FEATURE_COLS))
    assert y.shape == (6,)
    assert groups.shape == (6,)
    assert set(y) == {0, 1}


def test_build_subject_level_binary_encoding():
    df = _make_probe_df(n_subjects=6, probes_per_subject=4)
    X, y, groups = build_subject_level_features(df, FEATURE_COLS, "group", "Risk of Depression")
    assert y.sum() == 2


def test_build_probe_level_shapes():
    df = _make_probe_df(n_subjects=6, probes_per_subject=4)
    X, y, groups = build_probe_level_features(df, FEATURE_COLS, "group", "Risk of Depression")
    n_probes = 6 * 4 * 4  # subjects × tasks × probes_per_subject
    assert X.shape == (n_probes, len(FEATURE_COLS))
    assert y.shape == (n_probes,)
    assert groups.shape == (n_probes,)


def test_build_block_level_shapes():
    df = _make_probe_df(n_subjects=6, probes_per_subject=4)
    df_ie = df[df["inclusion_exclusion"].isin(["inclusion", "exclusion"])]
    X, y, groups = build_block_level_features(df_ie, FEATURE_COLS, "inclusion_exclusion", "inclusion")
    assert X.shape[0] == 6 * 2  # 6 subjects × 2 intervention blocks
    assert X.shape[1] == len(FEATURE_COLS)
    assert set(y) == {0, 1}


def test_build_block_level_groups_are_subjects():
    df = _make_probe_df(n_subjects=6, probes_per_subject=4)
    df_ie = df[df["inclusion_exclusion"].isin(["inclusion", "exclusion"])]
    X, y, groups = build_block_level_features(df_ie, FEATURE_COLS, "inclusion_exclusion", "inclusion")
    unique, counts = np.unique(groups.astype(str), return_counts=True)
    assert np.all(counts == 2)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import build_model, run_loso


def test_build_model_lr(config):
    model = build_model("lr", config)
    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
    assert model.random_state == config["models"]["random_state"]


def test_build_model_rf(config):
    model = build_model("rf", config)
    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.n_estimators == config["models"]["rf_n_estimators"]


def test_run_loso_returns_expected_keys():
    """LOSO on perfectly separable synthetic data returns valid AUC."""
    rng = np.random.RandomState(0)
    n_subjects = 10
    n_per_subject = 8
    X_list, y_list, g_list = [], [], []
    for s in range(n_subjects):
        label = 0 if s < 5 else 1
        X_subj = rng.randn(n_per_subject, 4)
        X_subj[:, 0] += label * 5.0
        X_list.append(X_subj)
        y_list.extend([label] * n_per_subject)
        g_list.extend([str(s)] * n_per_subject)
    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(g_list)

    config_stub = {
        "models": {"random_state": 0, "lr_C": 1.0, "lr_penalty": "l2",
                   "lr_max_iter": 500, "rf_n_estimators": 20, "rf_max_depth": 3},
        "loso": {"n_jobs": 1}
    }
    model = build_model("lr", config_stub)
    results = run_loso(X, y, groups, model, n_jobs=1)
    assert set(results.keys()) >= {"auc", "balanced_accuracy", "y_true", "y_proba",
                                    "subject_metrics", "feature_importances"}
    assert results["auc"] > 0.8
    assert 0.0 <= results["balanced_accuracy"] <= 1.0
    assert len(results["y_true"]) == len(y)
    assert results["feature_importances"].shape == (4,)


def test_run_loso_subject_level_single_sample_per_fold():
    """LOSO with 1 sample per subject (subject-level mode) aggregates AUC correctly."""
    rng = np.random.RandomState(1)
    n_subjects = 10
    X = rng.randn(n_subjects, 3)
    X[:5, 0] += 5.0
    y = np.array([0] * 5 + [1] * 5)
    groups = np.array([str(i) for i in range(n_subjects)])

    config_stub = {
        "models": {"random_state": 1, "lr_C": 1.0, "lr_penalty": "l2",
                   "lr_max_iter": 500, "rf_n_estimators": 20, "rf_max_depth": 3},
        "loso": {"n_jobs": 1}
    }
    model = build_model("rf", config_stub)
    results = run_loso(X, y, groups, model, n_jobs=1)
    assert 0.0 <= results["auc"] <= 1.0
    assert len(results["y_true"]) == n_subjects
